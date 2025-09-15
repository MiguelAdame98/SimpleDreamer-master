
def attrdict_monkeypatch_fix():
    import collections
    import collections.abc
    for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
attrdict_monkeypatch_fix()

import os
import os, datetime, pathlib, yaml, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from attrdict import AttrDict


def new_run_dir(base="./runs", exp_name="minigrid"):
    ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = pathlib.Path(base) / exp_name / ts
    path.mkdir(parents=True, exist_ok=False)
    (path / "ckpt").mkdir()
    return path

def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


class DynamicInfos:
    def __init__(self, device):
        self.device = device
        self.data = {}

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get_stacked(self, time_axis=1):
        stacked_data = AttrDict(
            {
                key: torch.stack(self.data[key], dim=time_axis).to(self.device)
                for key in self.data
            }
        )
        self.clear()
        return stacked_data

    def clear(self):
        self.data = {}


def find_file(file_name: str) -> str:
    """
    Walk downward from the current working directory until we find *file_name*.

    If *file_name* contains path components (e.g. 'configs/foo.yml') we ignore
    them and match only the basename, so either of the following now works:

        find_file('configs/minigrid-default.yml')
        find_file('minigrid-default.yml')
    """
    cur_dir = os.getcwd()
    base_name = os.path.basename(file_name)   # <-- strip any leading path

    for root, dirs, files in os.walk(cur_dir):
        if base_name in files:                # compare with the stripped name
            return os.path.join(root, base_name)

    raise FileNotFoundError(
        f"File '{file_name}' not found in subdirectories of {cur_dir}"
    )

def get_base_directory():
    return "/".join(find_file("main.py").split("/")[:-1])


'''def load_config(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = find_file(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)'''

import yaml, pprint
from pathlib import Path


def _merge_dicts(a: dict, b: dict) -> dict:
    """Recursively merge two dicts.  Keys in *b* override keys in *a*."""
    merged = dict(a)
    for k, v in b.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged

def _to_attrdict(d):
    """Turn nested dicts into dot-accessible namespaces."""
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_to_attrdict(x) for x in d]
    else:
        return d

def load_config(cfg_path: str):
    """
    Load YAML at *cfg_path*.
    If it contains a key `base_config: some_file.yml` we load that first
    and let the *current* file override it.
    Returns an AttrDict exactly like the original helper did.
    """

    cfg_path = Path(cfg_path).expanduser()
    if not cfg_path.suffix:
        cfg_path = cfg_path.with_suffix(".yml")
    assert cfg_path.exists(), f"Config file not found: {cfg_path}"

    # -------- 1. load the run-specific YAML ---------------------------------
    with cfg_path.open() as f:
        yaml_cfg = yaml.safe_load(f)

    # -------- 2. optionally load a base config ------------------------------
    base_cfg = {}
    if "base_config" in yaml_cfg:
        base_path = (cfg_path.parent / yaml_cfg["base_config"]).resolve()
        with base_path.open() as f:
            base_cfg = yaml.safe_load(f)

    # -------- 3. merge so YAML-file values override base --------------------
    merged_cfg = _merge_dicts(base_cfg, yaml_cfg)

    # -------- 4. debug prints ----------------------------------------------
    try:
        seed_nested = (
            merged_cfg["parameters"]["dreamer"]["seed_episodes"]
        )
    except KeyError:
        seed_nested = "N/A"

    print("\n[Debug-cfg] loaded YAML from", cfg_path)
    print("[Debug-cfg] seed_episodes after merge →", seed_nested, "\n")

    # -------- 5. return as AttrDict (dot-style access) ----------------------
    return _to_attrdict(merged_cfg)
