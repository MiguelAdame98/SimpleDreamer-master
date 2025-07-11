# ---------- main.py (after the imports) ---------------------------------
from pathlib import Path
from datetime import datetime
import shutil
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"

from torch.utils.tensorboard import SummaryWriter
from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, new_run_dir
from dreamer.envs.envs import (
    make_atari_env, make_minigrid_env, get_env_infos
)


def main(config_file: str | None, run_dir_arg: str | None):
    """
    If `run_dir_arg` is given we resume; otherwise we start a fresh run that
    stores its checkpoints in a new timestamped directory.
    """

    # ── 1. Choose the run directory ────────────────────────────────────
    if run_dir_arg:                                  # → resume
        run_dir = Path(run_dir_arg).expanduser()
        assert run_dir.exists(), f"{run_dir} does not exist"
        config_file = run_dir / "config.yml"         # use stored config
        print(f"[Debug] using run_dir’s config: {config_file}")
    else:                                            # → fresh run
        assert config_file, "--config is required when not resuming"
        run_dir = new_run_dir(exp_name="mg_collision")
        shutil.copy(config_file, run_dir / "config.yml")
        print(f"[Debug] copying  {config_file}  →  {run_dir/'config.yml'}")
    # ------------------------------------------------------------------
    print("\n[Debug] ────────────────────────────────────────────────")
    print(f"[Debug] CLI  --config  = {config_file}")
    print(f"[Debug] CLI  --run_dir = {run_dir_arg}")
    # ------------------------------------------------------------------
    # ── 2. Load config (now guaranteed to match the checkpoints) ───────
    config = load_config(str(config_file))
    print("[Debug] seed_episodes (root)                 =", getattr(config, "seed_episodes", None))
    print("[Debug] seed_episodes in config.dreamer      =", getattr(getattr(config, "dreamer", None), "seed_episodes", None))
    print("[Debug] seed_episodes in parameters.dreamer  =",
        getattr(config.parameters.dreamer, "seed_episodes", None))
    # ── 3. Make environment ────────────────────────────────────────────
    if config.environment.benchmark == "minigrid":
        env = make_minigrid_env(
            task_name  = config.environment.task_name,
            rooms_row  = config.environment.rooms_row,
            rooms_col  = config.environment.rooms_col,
            frame_skip = config.environment.frame_skip,
            pixel_norm = config.environment.pixel_norm,
            run_dir    = run_dir
        )
    else:
        raise ValueError(f"Unknown benchmark: {config.environment.benchmark}")

    obs_shape, discrete_action, action_size = get_env_infos(env)
    writer   = SummaryWriter(run_dir)
    device   = config.operation.device

    # ── 4. Instantiate algorithm ───────────────────────────────────────
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action, action_size,
            writer, device, config, run_dir
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action, action_size,
            writer, device, config
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # ── 5. Train / resume training ─────────────────────────────────────
    agent.train(env)


# ---------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dreamer/configs/minigrid-default.yml",
        help="Path to YAML config file (ignored when --run_dir is given)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run directory to resume (e.g. runs/mg_collision/20250704-220917)",
    )
    args = parser.parse_args()
    main(args.config, args.run_dir)
# -----------------------------------------------------------------------
'''import os, argparse
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path          #  ←  add this import
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory, new_run_dir

# ⬇️  NEW: include make_minigrid_env
from dreamer.envs.envs import (
    make_atari_env, make_minigrid_env, get_env_infos
)


def main(config_file):
    config = load_config(config_file)
    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    run_dir = new_run_dir(exp_name="mg_collision")
    shutil.copy(config_file, run_dir / "config.yml")   # save exact cfg

def main(config_file, run_dir_arg):
    # ① choose run directory
    if run_dir_arg is not None:                           # ← resume path supplied
        run_dir = Path(run_dir_arg).expanduser()
        assert run_dir.exists(), f"{run_dir} does not exist"
        config_file = run_dir / "config.yml"              # re-load the *saved* cfg
    else:                                                 # ← fresh run
        run_dir = new_run_dir(exp_name="mg_collision")    # makes YYMMDD-HHMMSS dir
        shutil.copy(config_file, run_dir / "config.yml")

    # ② load config (now guaranteed to match the checkpoints)
    config = load_config(str(config_file))



    writer   = SummaryWriter(run_dir)
    device = config.operation.device

    if config.environment.benchmark == "minigrid":          # ⬅️ NEW
        env = make_minigrid_env(
            task_name   = config.environment.task_name,
            rooms_row   = config.environment.rooms_row,
            rooms_col   = config.environment.rooms_col,
            frame_skip  = config.environment.frame_skip,
            pixel_norm  = config.environment.pixel_norm,
            run_dir=run_dir

        )
    else:
        raise ValueError(f"Unknown benchmark: {config.environment.benchmark}")
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)
    print(env.action_space)

    
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config,run_dir
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    agent.train(env)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Existing run directory to resume (e.g. runs/mg_collision/20250704-220917)",)
    args = parser.parse_args()
    main(args.config, args.run_dir)'''