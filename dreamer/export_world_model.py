import argparse, torch, importlib, pathlib

p = argparse.ArgumentParser()
p.add_argument("ckpt",  help="path/to/iterXXXXX.pt")
p.add_argument("--no-decoder",  action="store_true")
args = p.parse_args()

Device = "cpu"        # export on CPU

# ------------------------------------------------------------------ #
# 1)  load config + construct *matching* modules
ckpt = torch.load(args.ckpt, map_location=Device)
Dreamer = importlib.import_module("dreamer.algorithms.dreamer").Dreamer

# you need a config instance identical to training time
from dreamer.utils.utils import load_config
cfg = load_config("configs/minigrid-default.yml")       # or use ckpt["cfg"]

dummy = Dreamer((3,64,64), True, 3,            # observation shape, …
                writer=None, device=Device,
                config=cfg, run_dir=None)      # ← creates fresh modules

wanted = {"encoder", "rssm"}
if not args.no_decoder:
    wanted.add("decoder")

export = torch.nn.ModuleDict({k: Dreamer._modules(dummy)[k] for k in wanted})
for k in wanted:
    export[k].load_state_dict(ckpt["modules"][k])
export.eval()

out = pathlib.Path(args.ckpt).with_suffix(f"_{'-'.join(sorted(wanted))}.pt")
torch.save(export.state_dict(), out)
print("exported →", out)