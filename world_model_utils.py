# dreamer_mg/world_model_utils.py
# ----------------------------------------------------------------------
#  ✧  World-model utilities for MiniGrid collision prediction & planning
# ----------------------------------------------------------------------
import importlib, sys, pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from importlib import util
# or
from importlib.util import spec_from_file_location, module_from_spec

# point "dreamer" to dreamer_mg.dreamer
pkg_path = pathlib.Path(__file__).parent / "dreamer"
spec = importlib.util.spec_from_file_location("dreamer", pkg_path / "__init__.py")
dreamer_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dreamer_pkg)
sys.modules["dreamer"] = dreamer_pkg
import yaml, torch, pathlib, heapq, random, numpy as np
from collections import namedtuple
from types import SimpleNamespace  
from dreamer.modules.encoder import Encoder
from dreamer.modules.model   import RSSM
from dreamer.modules.decoder import Decoder 
from dreamer.modules.model import RewardModel
from attrdict import AttrDict 
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym.wrappers import ResizeObservation
from dreamer.envs.wrappers import ChannelFirstEnv
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import textwrap, pprint, itertools
import math
from PIL import Image, ImageDraw, ImageFont   # Pillow
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils
import torchvision.transforms.functional as TF 
from torchvision.utils import save_image
from pathlib import Path
from itertools import islice
from torchvision.utils import save_image
 # --- Cognitive graph harness ---------------------------------------------------
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import sys as _sys
import math 


_REPO_ROOT = str(Path(__file__).resolve().parents[1])  # one level up from dreamer_mg/
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# Optional deps used in the template encoder; all guarded.
try:
    import cv2  # for resize / gradients if available
except Exception:
    cv2 = None
try:
    import yaml
except Exception:
    yaml = None

# Robust imports for your MemoryGraph + config loader
MemoryGraph = None
setup_memory_config = None
_import_errs = []
for modpath, name in [
    ("navigation_model.Services.memory_service.memory_graph", "MemoryGraph"),
    ("navigation_model.Services.memory_service.memory_graph.memory_graph", "MemoryGraph"),
    ("navigation_model.Services.memory_service", "memory_graph"),
]:
    try:
        mod = __import__(modpath, fromlist=[name])
        MemoryGraph = getattr(mod, name) if hasattr(mod, name) else getattr(mod, "MemoryGraph")
        break
    except Exception as e:
        _import_errs.append((modpath, str(e)))

if setup_memory_config is None:
    for modpath in [
        "control_eval.input_output",
        "navigation_model.control_eval.input_output",
        "control_eval.io",
    ]:
        try:
            mod = __import__(modpath, fromlist=["setup_memory_config"])
            setup_memory_config = getattr(mod, "setup_memory_config")
            break
        except Exception as e:
            _import_errs.append((modpath, str(e)))

@dataclass
class CognitiveGraphHarness:
    """
    Minimal helper around MemoryGraph so the planner can update/read the 'cognitive graph'
    without importing your old Manager.
    """
    cfg_path: Optional[str] = None
    cfg_obj: Optional[Dict[str, Any]] = None
    mg: Any = None  # MemoryGraph instance
    _use_manager_exact_template: bool = False  # if True, plug your original rgb56_to_template64

    def __post_init__(self):
        if MemoryGraph is None:
            raise ImportError(f"Could not import MemoryGraph. Tried: {_import_errs}")
        if self.cfg_obj is None:
            if setup_memory_config is not None and self.cfg_path:
                self.cfg_obj = setup_memory_config(self.cfg_path)
            else:
                if not (self.cfg_path and yaml):
                    raise ValueError("Provide cfg_path (and have PyYAML available) or pass cfg_obj directly.")
                with open(self.cfg_path, "r") as f:
                    self.cfg_obj = yaml.safe_load(f)
        print("THIS IS THE CFG OBJ",self.cfg_obj)
        self.mg = MemoryGraph(**self.cfg_obj)

    # ---------- image → 64D template ----------
    def _to_template64(self, image_rgb: np.ndarray) -> "np.ndarray | torch.Tensor":
        """
        Safe, dependency-light 64D descriptor:
        - resize to 56x56
        - grayscale
        - average-pool to 8x8 → 64 dims
        Replace this with your exact rgb56_to_template64 for full fidelity.
        """
        arr = np.asarray(image_rgb)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if cv2 is not None:
                img = cv2.resize(arr, (56, 56), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                # naive fallback
                from math import floor
                # quick nearest-neighbor shrink to 56 if needed
                if arr.shape[0] != 56 or arr.shape[1] != 56:
                    # crude resize with slicing
                    step0 = max(1, arr.shape[0] // 56)
                    step1 = max(1, arr.shape[1] // 56)
                    img = arr[::step0, ::step1, :]
                    img = img[:56, :56, :]
                else:
                    img = arr
                gray = (0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]).astype(np.float32)
            # 8x8 average pooling
            pool = gray.reshape(8, 7, 8, 7).mean(axis=(1,3))  # (8,8)
            vec64 = (pool / (pool.sum() + 1e-8)).reshape(-1).astype(np.float32)
            return vec64
        raise ValueError("Expected HxWx3 RGB image")

    # ---------- public API ----------
    def digest(
        self,
        image_rgb: np.ndarray,
        action_onehot: np.ndarray,        # [F,R,L] one-hot or similar
        pose_xyz: Tuple[int, int, int],   # (x,y,th) with th∈{0,1,2,3}
        *,
        place_post: Optional["np.ndarray | Any"] = None,  # optional (allocentric posterior)
        place_std: Optional[float] = None,
        std_th: Optional[float] = None,
        agent_lost: bool = False,
    ):
        obs = {
            "image": self._to_template64(image_rgb),
            "HEaction": np.asarray(action_onehot, dtype=np.float32),
            "pose": np.asarray(pose_xyz, dtype=np.int32),
            "place": None,
        }
        if not agent_lost and place_post is not None and (place_std is None or (std_th is not None and place_std < std_th)):
            # Mirror your manager: mean over posterior samples along dim 0
            try:
                import torch
                if isinstance(place_post, torch.Tensor):
                    obs["place"] = place_post.mean(dim=0).squeeze(0)
                else:
                    # assume numpy: mean over axis 0
                    obs["place"] = np.mean(place_post, axis=0)
            except Exception:
                obs["place"] = None
        else:
            # reset 'door memories' if lost (no-op if your MemoryGraph ignores this)
            if hasattr(self.mg, "memorise_poses"):
                try:
                    self.mg.memorise_poses([])
                except Exception:
                    pass
        self.mg.digest(obs, dt=1, adjust_map=False)

    def nodes(self) -> List[Dict[str, Any]]:
        """Return a lightweight summary of experience nodes & links."""
        out = []
        emap = getattr(self.mg, "experience_map", None)
        if emap is None:
            return out
        exps = getattr(emap, "exps", [])
        for e in exps:
            # try to be permissive about field names
            x = getattr(e, "x_m", getattr(e, "x", 0.0))
            y = getattr(e, "y_m", getattr(e, "y", 0.0))
            eid = getattr(e, "id", None)
            links = []
            for lk in getattr(e, "links", []):
                tid = getattr(getattr(lk, "target", None), "id", None)
                if tid is not None:
                    links.append(tid)
            out.append({"id": eid, "x": float(x), "y": float(y), "links": links})
        return out

    def node_positions(self) -> List[Tuple[float, float]]:
        return [(n["x"], n["y"]) for n in self.nodes()]


# ----------------------------------------------------------------------
#  ✧ DualPathCollage: build two separate collages (REAL vs IMAGINED)
#     Call .add_step(...) every step; call .finalize() at the end.
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import math
import numpy as np

try:
    import torch
    import torchvision.transforms.functional as TF
    import torchvision.utils as vutils
except Exception as _e:
    torch = None
    TF = None
    vutils = None

@dataclass
class DualPathCollage:
    out_dir: str = "dbg/collages"
    tag: str = "run"
    tile: int = 64                   # tile edge in pixels (images are resized to tile x tile)
    cols: int = 20                   # columns in the collage grid
    live_write: bool = False         # if True, re-save collages after every add_step
    live_every: int = 1              # save every N steps when live_write=True
    pad: int = 1                     # padding between tiles (in pixels)
    pad_value: float = 1.0           # 1.0 -> white padding
    annotate_idx: bool = False       # draw step indices on each tile (requires PIL)
    _reals: List[Any] = field(default_factory=list, init=False, repr=False)
    _imags: List[Any] = field(default_factory=list, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    # Optional hooks if you want the class to decode for you
    planner: Any = None
    wm: Any = None

    def _ensure_deps(self):
        if torch is None or vutils is None or TF is None:
            raise RuntimeError(
                "DualPathCollage requires torch/torchvision. "
                "Please ensure they are available in this environment."
            )

    @staticmethod
    def _to_chw01(x) -> "torch.Tensor":
        """Accept np.uint8 HWC, torch CHW or HWC, ranges [0,1] or [-.5,.5] → CHW float in [0,1]."""
        # torch already?
        if 'torch' in str(type(x)):
            t = x.detach().float().cpu()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            # HWC → CHW
            if t.ndim == 3 and t.shape[-1] in (1,3) and t.shape[0] not in (1,3):
                t = t.permute(2,0,1).contiguous()
            # single-channel → 3-channel
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.repeat(3,1,1)
            # normalize to [0,1]
            if t.min() < 0.0 or t.max() <= 1.0 and t.min() >= -0.6:
                # assume Dreamer-style [-.5,.5] or [0,1]
                t = (t + 0.5).clamp(0.0, 1.0)
            else:
                t = t.clamp(0.0, 1.0)
            return t

        # numpy path
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.dtype == np.uint8:
                # HWC uint8 [0..255]
                t = torch.as_tensor(arr).permute(2,0,1).float().cpu() / 255.0
            elif arr.ndim == 3:  # float?
                t = torch.as_tensor(arr).permute(2,0,1).float().cpu()
                if t.min() < 0.0 or (t.max() <= 1.0 and t.min() >= -0.6):
                    t = (t + 0.5).clamp(0.0, 1.0)
                else:
                    t = t.clamp(0.0, 1.0)
            else:
                raise ValueError(f"Unsupported numpy array shape: {arr.shape}")
            if t.shape[0] == 1:
                t = t.repeat(3,1,1)
            return t

        raise ValueError(f"Unsupported image type: {type(x)}")

    @staticmethod
    def _resize_to_tile(chw01: "torch.Tensor", tile: int) -> "torch.Tensor":
        """Resize CHW [0,1] to tile×tile using nearest to keep MiniGrid pixels crisp."""
        if TF is None:
            return chw01
        return TF.resize(chw01, [tile, tile], interpolation=TF.InterpolationMode.NEAREST)

    def _maybe_annotate(self, chw01: "torch.Tensor", idx: int) -> "torch.Tensor":
        if not self.annotate_idx:
            return chw01
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil = TF.to_pil_image(chw01)
            draw = ImageDraw.Draw(pil)
            # tiny label in the corner; default font to avoid platform issues
            draw.rectangle([0,0,16,12], fill=(255,255,255))
            draw.text((2,0), str(idx), fill=(0,0,0))
            return TF.to_tensor(pil)
        except Exception:
            return chw01

    def add_step(
        self,
        real_img: Any = None,
        imagined_img: Any = None,
        belief_zd: Optional[Tuple[Any,Any]] = None
    ):
        """
        Add one step to the collages.
        - Pass `real_img` from env (e.g., obs["image"])
        - EITHER pass `imagined_img`, OR pass `belief_zd` and set planner/wm in the constructor.
        """
        self._ensure_deps()
        self._step += 1

        if real_img is None:
            raise ValueError("add_step requires real_img")

        # Prepare REAL
        r = self._to_chw01(real_img)
        r = self._resize_to_tile(r, self.tile)
        r = self._maybe_annotate(r, self._step)
        self._reals.append(r)

        # Prepare IMAGINED
        im = imagined_img
        if im is None and belief_zd is not None and self.planner is not None and self.wm is not None:
            # Reuse your existing utility if available in this module:
            try:
                # dreamer_decode_from_belief is defined elsewhere in this file
                im = dreamer_decode_from_belief(self.planner, self.wm, belief_zd, to_01=True)
            except Exception:
                im = None
        if im is None:
            # Fallback: if we can't decode, mirror the real image so can still visualize layout
            im = r
        else:
            im = self._to_chw01(im)
            im = self._resize_to_tile(im, self.tile)
            im = self._maybe_annotate(im, self._step)
        self._imags.append(im)

        if self.live_write and (self._step % self.live_every == 0):
            self._save_collage(pair=("real", "imagined"))

    def _save_collage(self, pair=("real","imagined")):
        """Save current collages to out_dir as two separate PNGs."""
        import os
        from pathlib import Path
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        def write_one(stack_list: List["torch.Tensor"], suffix: str):
            if not stack_list:
                return
            grid = vutils.make_grid(
                torch.stack(stack_list, dim=0),
                nrow=self.cols,
                padding=self.pad,
                pad_value=self.pad_value
            )
            out_path = os.path.join(self.out_dir, f"{self.tag}_{suffix}.png")
            vutils.save_image(grid, out_path)
            return out_path

        real_path = write_one(self._reals, "real")
        imag_path = write_one(self._imags, "imagined")
        return real_path, imag_path

    def finalize(self):
        """Write the final collages for REAL and IMAGINED."""
        return self._save_collage(pair=("real","imagined"))


# ─────────────────────────────── constants ────────────────────────────
State    = namedtuple("State", ["x", "y", "d"])          # planner state
DIR_VECS = [(1,0), (0,1), (-1,0), (0,-1)]                # 0:right 1:down …

DEVICE   = "cpu"

class WMPlanner:
    """High-level wrapper: loading, belief update, collision check, A* and render."""
    # ──────────────────────────────────────────────────────────────
    def __init__(self, ckpt: str, cfg: str | None = None, device="cpu",
                 memory_config: str | None = None, memory_cfg_obj: Dict[str, Any] | None = None):
        self.wm = self.load_world_model2(ckpt_path=ckpt)
        self.wm.device = device
        self.device = device
        self.cog: Optional[CognitiveGraphHarness] = None
        self.env=None
        if memory_config or memory_cfg_obj:
            self.cog = CognitiveGraphHarness(cfg_path=memory_config, cfg_obj=memory_cfg_obj)
        
        emap = self.cog.mg.experience_map
        self.emap=emap
    def load_world_model2(self,ckpt_path: str,
                        config_yaml: str = "/Users/lab25/hierarchical-nav/dreamer_mg/runs/mg_collision/20250704-220917/config.yml"):


        """
        Returns an object with .encoder and .rssm that exactly match *any*
        Dreamer checkpoint – no manual YAML tweaks required.

        • Reads sizes directly from the ckpt:
            in_dim  = W.shape[1]  of recurrent_model.linear.weight
            stoch   = (W₂.shape[0]) // 2      (# outputs / 2)
            act_sz  = in_dim - stoch
        • Injects the discovered numbers into a *copy* of the yaml dict.
        """
        import copy, yaml, torch
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

            # --- 1. discover sizes ------------------------------------------------
        Wrec   = ckpt["modules"]["rssm"]["recurrent_model.linear.weight"]
        H3, in_dim = Wrec.shape                 # (3H, input)
        hidden = H3 // 3                        # true deterministic_size
        Wtrans = ckpt["modules"]["rssm"]["transition_model.network.2.weight"]
        stoch  = Wtrans.shape[0] // 2
        act_size= in_dim - stoch

        # --- 2. patch the yaml ------------------------------------------------
        cfg = AttrDict(yaml.safe_load(open(config_yaml)))
        cfg.parameters.dreamer.deterministic_size = hidden
        cfg.parameters.dreamer.stochastic_size    = stoch
        #cfg['parameters']['dreamer']['deterministic_size'] = H
        #cfg['parameters']['dreamer']['stochastic_size']    = stoch_sz

        enc  = Encoder((3, 64, 64), cfg).to(DEVICE)
        dec = Decoder((3, 64, 64), cfg).to(DEVICE)
        rssm = RSSM(action_size=act_size, config=cfg).to(DEVICE)
        rew  = RewardModel(config=cfg).to(DEVICE)

        enc .load_state_dict(ckpt["modules"]["encoder"]);  enc .eval()
        rssm.load_state_dict(ckpt["modules"]["rssm"   ]);  rssm.eval()
        dec.load_state_dict(ckpt["modules"]["decoder"]); dec.eval()
        rew.load_state_dict(ckpt["modules"]["reward"]); rew.eval()

        wm = lambda: None
        wm.encoder = enc
        wm.rssm    = rssm
        wm.decoder = dec
        wm.action_size = act_size                        # handy later
        wm.reward_predictor = rew
        wm.device=DEVICE
        if act_size == 3:                               # NEW model
            wm.idx = {"left": 0, "right": 1, "forward": 2}
        elif act_size == 7:                             # old checkpoints
            wm.idx = {
                "left":0,"right":1,"forward":2,
                "pickup":3,"drop":4,"toggle":5,"done":6}
        else:
            raise ValueError(f"Unexpected action dim {act_size}")
        print("rssm expects action dim:", wm.action_size)
        return wm
    def onehot(self,a_name: str,wm):
        if a_name not in wm.idx:
            raise KeyError(f"action '{a_name}' not in mapping {wm.idx}")
        v = torch.zeros(wm.action_size, device=DEVICE)
        v[wm.idx[a_name]] = 1.0
        return v

    # ══════════════════════════════════════════════════════════════════════
    # 2.  COLLISION PREDICTION  (wm_predict_collision)
    # ══════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    @torch.no_grad()
    def wm_update_belief(self,wm,
                        prev_z_d: tuple[torch.Tensor, torch.Tensor],
                        frame_rgb: np.ndarray,
                        prev_action_onehot: torch.Tensor | None):
        """
        Update (z,d) belief given *one* observation & the *previous* action.

        Parameters
        ----------
        prev_z_d          tuple(z,d) from the previous step OR  None on first step
        frame_rgb         np.ndarray  (3,64,64)   current observation
        prev_action_onehot  torch.Tensor shape (1,3)  one-hot of action_t-1
                            →  pass None right after reset()

        Returns
        -------
        z_t, d_t  :  the new posterior latent tensors (no grads)
        """
        if prev_z_d is None:                               # first frame
            # dummy recurrent input (batch=1)
            prior, det = wm.rssm.recurrent_model_input_init(1)
        else:
            prior, det = prev_z_d

        # recurrent model if we have an *action that already happened*
        if prior is not None and prev_action_onehot is not None:
            det = wm.rssm.recurrent_model(
                prior,
                prev_action_onehot.unsqueeze(0),  # (1, action_size)
                det)
        prior_dist, prior = wm.rssm.transition_model(det)
        # encode observation & run representation model
        img = torch.tensor(frame_rgb, dtype=torch.float32,
                        device=DEVICE).unsqueeze(0)
        emb = wm.encoder(img).view(1, -1)
        _, post = wm.rssm.representation_model(emb, det)   # posterior 

        return post.detach(), det.detach()
   

    def _save_rollout_grid(self,frames, rewards, tag, upscale=256):
        """
        frames  : list[Tensor] each 3×64×64, values in [-.5,.5] or [0,1]
        rewards : list[float]  same length (use float('nan') for “no reward”)
        tag     : str          filename suffix, e.g. '00', '01', ...
        """

        assert len(frames) == len(rewards), "frames and rewards length mismatch"
        Path("dbg").mkdir(exist_ok=True)

        pil_frames = []
        for fr in frames:
            # (3,64,64) → (64,64,3) uint8 in [0,255]
            fr = fr.clone().cpu()
            if fr.min() < 0:
                fr = fr + 0.5
            fr = (fr.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img = Image.fromarray(fr)
            img = img.resize((upscale, upscale), resample=Image.NEAREST)

            pil_frames.append(img)

        # build a wide canvas: W = T*upscale, H = upscale+20
        T = len(pil_frames)
        canvas = Image.new("RGB", (T * upscale, upscale + 20), color="white")

        # optional: nicer mono-space font; fall back to default if not found
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size=14)
        except IOError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)
        for i, (img, r) in enumerate(zip(pil_frames, rewards)):
            # paste frame
            canvas.paste(img, (i * upscale, 0))

            # reward text centred under the frame
            txt = "nan" if np.isnan(r) else f"{r:+.2f}"
            if hasattr(draw, "textbbox"):                 # Pillow ≥ 10
                x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
                w, h = x1 - x0, y1 - y0
            else:                                         # Pillow < 10
                w, h = draw.textsize(txt, font=font)
            x = i * upscale + (upscale - w) // 2
            y = upscale + (20 - h) // 2
            draw.text((x, y), txt, fill="black", font=font)

        fname = f"dbg/coll_chk_{tag}.png"
        canvas.save(fname)
        print(f"saved {fname}")
    @torch.no_grad()
    
    def save_front_patches(self,frame: torch.Tensor,
                        tag: str = "step",
                        agent_tile=(3, 6),
                        core_fraction: float = 2/3):
        """
        frame : 3×H×W tensor in [-.5,.5] or [0,1]
        Saves three PNGs under dbg/:
            dbg/<tag>_frame.png          – full decoded frame
            dbg/<tag>_patch_full.png     – 9×9 (or 18×18) full tile
            dbg/<tag>_patch_core.png     – centred sub-patch of that tile
        """
        Path("dbg").mkdir(exist_ok=True, parents=True)

        # put frame in [0,1] for saving
        vis = frame.detach().clone()
        if vis.min() < 0: vis += 0.5
        vis = vis.clamp(0, 1)

        # full 9×9 tile
        full_patch = self._front_patch(vis, agent_tile=agent_tile, core_fraction=1.0)

        # centre crop
        core_patch = self._front_patch(vis, agent_tile=agent_tile,
                                core_fraction=core_fraction)
        p = core_patch.detach().cpu().clamp(0,1)
        print("mean RGB :", p.mean(dim=(1,2)))
        print("variance :", p.var())
        up = vis.shape[-1]                 # 64 (or 128, …)
        full_up = F.interpolate(full_patch.unsqueeze(0), size=vis.shape[-1],
                                mode="nearest").squeeze(0)
        core_up = F.interpolate(core_patch.unsqueeze(0), size=vis.shape[-1],
                                mode="nearest").squeeze(0)

        imgs = torch.stack([vis, full_up, core_up], dim=0)   # (3, 3, 64, 64)

        save_image(imgs,
                f"dbg/{tag}_triplet.png",
                nrow=3,          # one row: frame | full | core
                normalize=True, scale_each=False)
        print(f"saved dbg/{tag}_triplet.png")
        print("bright :", (core_patch.mean()).item())
        print("sat    :", (core_patch[0]-core_patch[1]).abs().max().item())
        print("var    :", core_patch.var().item())
        save_image(vis,         f"dbg/{tag}_frame.png",  normalize=True)
        save_image(full_patch,  f"dbg/{tag}_patch_full.png", normalize=True)
        save_image(core_patch,  f"dbg/{tag}_patch_core.png", normalize=True)

        print(f"saved dbg/{tag}_frame.png, ..._patch_full.png, ..._patch_core.png")
    
    def _front_patch(self,frame: torch.Tensor,
                    agent_tile=(3, 6),
                    core_fraction: float = 2/3):
        H   = frame.shape[-1]            # assume square
        tile = H // 7                    # 9 for 64×64, 18 for 128×128 …
        margin = (H - tile * 7) // 2     # 1-px left/top margin at 64×64
        fx, fy = agent_tile[0], agent_tile[1] - 1   # tile straight ahead

        # full front-tile coordinates
        x0 = margin + fx * tile
        y0 = margin + fy * tile
        full_tile = frame[:, y0:y0+tile, x0:x0+tile]   # 3×tile×tile

        if not (0 < core_fraction < 1):
            return full_tile                           # return whole tile

        # size of the centred crop
        c = int(round(tile * core_fraction))           # e.g. 6
        core_tile = TF.center_crop(full_tile, (c, c)) 
        return core_tile

    def _is_wall_patch(self,patch: torch.Tensor,
                    bright_min: float = 0.35,
                    sat_max:   float = 0.04,
                    frac_min:  float = 0.60) -> bool:
        """
        Treat the patch as a wall if at least `frac_min` of its pixels are
        both bright AND grey (low saturation).

        bright := (R+G+B)/3 > bright_min
        grey   := max(|R-G|,|G-B|,|B-R|) < sat_max
        """
        p = patch.detach().cpu().clamp(0, 1)              # 3×T×T  in [0,1]
        R, G, B = p                                       # unpack channels
        brightness = (R + G + B) / 3                      # T×T
        saturation = torch.max(torch.stack([
                            (R-G).abs(),
                            (G-B).abs(),
                            (B-R).abs()]), dim=0).values  # T×T

        mask = (brightness > bright_min) & (saturation < sat_max)
        frac = mask.float().mean().item()                 # fraction of “wall-like” px
        return frac >= frac_min

    

    def save_tiled_overlay(self,frame: torch.Tensor,
                        tag: str = "latest",
                        upscale: int = 256):
        """
        frame  : 3×H×W tensor in [-.5,.5] or [0,1]
        Writes dbg/overlay_<tag>.png  +  overlay_<tag>.npy (7×7×3 RGB means)
        """
        Path("dbg").mkdir(exist_ok=True, parents=True)
        png_name = f"dbg/overlay_{tag}.png"
        npy_name = f"dbg/overlay_{tag}.npy"

        H = frame.shape[-1]                       # assume square
        tile = H // 7
        margin = 3

        # move to [0,1] and PIL
        img = frame.detach().clone()
        if img.min() < 0: img = img + 0.5
        img = img.clamp(0,1)
        pil = vutils.make_grid(img.unsqueeze(0)).permute(1,2,0).numpy()*255
        pil = Image.fromarray(pil.astype("uint8"))
        pil = pil.resize((upscale,upscale), resample=Image.NEAREST)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()

        rgb_means = np.zeros((7,7,3), dtype=np.float32)

        # draw tiles
        up_tile = upscale // 7
        for y in range(7):
            for x in range(7):
                px0 = margin + x*tile
                py0 = margin + y*tile
                patch = img[:, py0:py0+tile, px0:px0+tile]
                rgb   = patch.mean(dim=(1,2)).cpu().numpy()
                rgb_means[y,x] = rgb

                # grid on big image
                gx0, gy0 = x*up_tile, y*up_tile
                draw.rectangle([gx0, gy0, gx0+up_tile, gy0+up_tile],
                            outline="white", width=1)

                # two-line label
                txt1 = f"{x},{y}"
                txt2 = f"{rgb[0]:+.2f},{rgb[1]:.2f},{rgb[2]:.2f}"
                draw.text((gx0+2, gy0+2), txt1, fill="yellow", font=font)
                draw.text((gx0+2, gy0+12), txt2, fill="white",  font=font)

        pil.save(png_name)
        np.save(npy_name, rgb_means)
        print(f"saved {png_name}  &  {npy_name}")

        # console table for quick glance
        for y in range(7):
            row = " | ".join(f"{rgb_means[y,x,0]:.2f},{rgb_means[y,x,1]:.2f},{rgb_means[y,x,2]:.2f}"
                            for x in range(7))
            print(f"y={y}  {row}")
    def wm_collision_by_decoder(
            self,
            wm,
            belief_zd: tuple[torch.Tensor, torch.Tensor],
            seq: list[str],
            wall_gray_thresh: float = 0.65,
            num_rollouts: int = 2,
            debug: bool = False,
    ) -> bool:
        """
        Look at the decoded image after each imagined action and declare UNSAFE
        if the front-tile patch is brighter than `wall_gray_thresh`.
        """
        z0, d0 = (x.detach() for x in belief_zd)
        idx  = torch.tensor([wm.idx[a] for a in seq], device=wm.device)
        oneh = F.one_hot(idx, num_classes=wm.action_size).float()
        unsafe = False
        # static attribute lives across calls
        if not hasattr(self.wm_collision_by_decoder, "_wall_id"):
            self.wm_collision_by_decoder._wall_id = 0

        for r in range(num_rollouts):
            z, d = z0.clone(), d0.clone()

            if debug:
                frames_dbg, wall_dbg = [], []

            for t, a in enumerate(oneh, 1):
                d = wm.rssm.recurrent_model(z, a.unsqueeze(0), d)
                _, z = wm.rssm.transition_model(d)
                frame = wm.decoder(z, d).mean.squeeze(0)        # 3×64×64, in [-.5,.5]
                vis = (frame + 0.5).clamp(0, 1)                 # → [0,1]
                patch = self._front_patch(vis)                       # 3×tile×tile
                #save_front_patches(frame, tag="step011")
                #save_tiled_overlay(vis, tag="_".join(seq))
                
                is_wall = self._is_wall_patch(patch)

                if debug:
                    frames_dbg.append(vis.cpu())
                    wall_dbg.append(is_wall)

                if is_wall:
                    wall_tag = f"wall_{self.wm_collision_by_decoder._wall_id:05d}"
                    self.save_front_patches(frame, tag=wall_tag)   # writes triplet
                    self.wm_collision_by_decoder._wall_id += 1
                    unsafe = True
                    break

            if debug:
                self._save_rollout_grid(frames_dbg,
                                [1.0 if w else 0.0 for w in wall_dbg],
                                tag=f"pix{self.wm_collision_by_decoder._counter:05d}")
                self.wm_collision_by_decoder._counter += 1

            if unsafe:
                break

        return unsafe

    # initialise the static counter
    wm_collision_by_decoder._counter = 0


    def is_wall_ahead_now(self,wm, belief_zd, debug=False, tag=None) -> bool:
        """
        Decode the *current* latent, grab the front core patch, and test it.
        No rollout at all.
        """
        z, d = belief_zd
        frame = wm.decoder(z, d).mean.squeeze(0)   # [0,1]
        vis = (frame + 0.5).clamp(0, 1) 
        patch = self._front_patch(vis)                      # 6×6 by default
        wall  = self._is_wall_patch(patch)

        if debug and wall and tag is not None:
            self.save_front_patches(frame, tag=tag)           # optional snapshot
        return wall
    # ─────────────────────────────────────────────────────────────────────
    # Decision tree node to expose per-node metrics and children
    # ─────────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
    # Decision tree node to expose per-node metrics and children
    # ─────────────────────────────────────────────────────────────────────
    @dataclass
    class TreeNode:
        node_id: int
        depth: int
        pose: Tuple[int, int, int]                  # (x,y,d)
        seq: List[str]                              # actions from root to this node
        belief_zd: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (z,h) DETACHED, on CPU
        # per-step metrics for edge (parent → this)
        step_act: Optional[str] = None
        step_novelty: float = 0.0
        step_penalty: float = 0.0
        step_p2e: float = 0.0
        step_graph: float = 0.0
        step_reward: float = 0.0                    # combined reward for that edge
        cum_gain: float = 0.0                       # discounted sum from root to this node
        children: List["WMPlanner.TreeNode"] = field(default_factory=list)
        is_leaf: bool = False
        p2e_leaf_value: Optional[float] = None      # cached K-step value at the node

    # ─────────────────────────────────────────────────────────────────────
    # Small helpers (latent stepping + self-disagreement)
    # ─────────────────────────────────────────────────────────────────────
    def rssm_step(self, wm, belief, act_name: str):
        """One Dreamer PRIOR step. Returns NEXT (z,h) with no grads."""
        import torch.nn.functional as F
        z, h = belief
        onehot = F.one_hot(
            torch.tensor([wm.idx[act_name]], device=wm.device),
            num_classes=wm.action_size
        ).float()
        with torch.no_grad():
            h_next = wm.rssm.recurrent_model(z, onehot, h)
            _, z_next = wm.rssm.transition_model(h_next)
        return (z_next.detach(), h_next.detach())
    def _prior_mean(self, dist) -> torch.Tensor:
        mu = getattr(dist, "mean", None)
        if mu is None:
            # last-resort: try loc (Normal(loc, scale))
            mu = getattr(dist, "loc", None)
        return torch.as_tensor(mu).float()

    def _pairwise_dispersion(self, vecs, metric: str = "l2") -> float:
        """Average pairwise distance across a list of 1D tensors (CPU)."""
        import torch
        V = [torch.as_tensor(v).view(-1).detach().cpu().float() for v in vecs]
        n = len(V)
        if n < 2:
            return 0.0
        total = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                if metric == "cos":
                    a, b = V[i], V[j]
                    denom = (a.norm() * b.norm()).clamp_min(1e-8)
                    d = 1.0 - float((a @ b) / denom)
                else:  # "l2"
                    d = float((V[i] - V[j]).pow(2).mean().sqrt())
                total += d
                cnt += 1
        return total / cnt
    def _roll_same_action_collect_z(self, wm, belief, act_name: str, K: int):
        """Roll K steps applying the same action; return list of z (flattened)."""
        zs = []
        cur = belief
        for _ in range(max(0, K)):
            cur = self.rssm_step(wm, cur, act_name)
            z, _h = cur
            zs.append(z.view(-1).detach().cpu().float())
        return zs
    def _roll_same_action_collect_means(self, wm, belief, act_name: str, K: int) -> List[torch.Tensor]:
        """
        Deterministic rollout K steps applying the same action; return list of prior means.
        """
        means = []
        cur = belief
        for _ in range(K):
            (cur, _prior) = self.rssm_step(wm, cur, act_name)
            # Note: rssm_step already sampled z_next; we also want the *mean* of that prior:
            # re-compute prior on the new (z,h) to get its mean reproducibly:
            _, prior_dist = wm.rssm.transition_model(cur[1])  # prior over current h
            mu = self._prior_mean(prior_dist).view(-1).cpu()
            means.append(mu)
        return means

    def _mc_dispersion_same_action(self, wm, belief, act_name: str, K: int, S: int, metric: str) -> float:
        """
        Monte Carlo self-disagreement:
        - For each future step t=1..K, sample S latents from the PRIOR over z_t
        under repeated action 'act_name'. Measure dispersion across samples.
        - Return the average dispersion across t.
        """
        if K <= 0 or S <= 1:
            return 0.0
        cur = belief
        step_vals = []
        with torch.no_grad():
            for _ in range(K):
                # one recurrent step to get the *next* prior
                (next_belief, prior_dist) = self.rssm_step(wm, cur, act_name)
                # sample S latents from that prior
                samples = []
                for _s in range(S):
                    try:
                        z_s = prior_dist.rsample()  # (1, Z)
                    except Exception:
                        z_s = prior_dist.sample()
                    samples.append(z_s.view(-1).cpu().float())
                # dispersion across samples at current step
                step_vals.append(self._pairwise_dispersion(samples, metric=metric))
                cur = next_belief
        return float(np.mean(step_vals)) if step_vals else 0.0

    def p2e_leaf_value(
        self,
        wm,
        belief,
        act_name: str,
        *,
        K: int = 4,
        metric: str = "l2",
        roll_from: str = "parent"  # "parent": start at node belief; "child": start after 1 step
    ) -> float:
        """
        K-rollout self-disagreement for a single action from a node:
        - If roll_from="parent": start at the node's belief, apply 'act_name' K times.
        - If roll_from="child" : first step to the child, then apply 'act_name' K times.
        - Compare the K predicted z's to each other (pairwise dispersion).
        """
        if K <= 0:
            return 0.0
        start_belief = belief
        if roll_from == "child":
            start_belief = self.rssm_step(wm, belief, act_name)
        zs = self._roll_same_action_collect_z(wm, start_belief, act_name, K)
        return self._pairwise_dispersion(zs, metric=metric)

    def open_graph_bonus(self, predicted_pose, memory_graph=None) -> float:
        """
        Hook for cognitive-graph term. For now this is stubbed to 0.0.
        Later: e.g., +1 if pose would create a new node or reduce distance to frontier.
        """
        return 0.0


    # ══════════════════════════════════════════════════════════════════════
    # 3.  A*  PLANNER  (astar_prims)  –  uses wm_predict_collision
    #     Changes:
    #       (A) forward-collision now uses MC votes at current belief
    #       (B) closed set is depth-aware (reopen stride) + g_score relaxed
    # ══════════════════════════════════════════════════════════════════════
    def heuristic(self, s: State, g: State) -> int:
        manh = abs(s.x - g.x) + abs(s.y - g.y)
        turn = min((s.d - g.d) % 4, (g.d - s.d) % 4)
        return manh + turn

    def astar_prims(self,
            wm,
            belief_zd,                  # (z,h) (latent + RNN state) at the *start* frame
            start: State,
            goal:  State,
            max_actions: int = 20,      # hard budget
            num_rollouts: int = 6,
            verbose: bool = False,
            allow_partial: bool = False):
        """
        A* over primitive actions with MC forward-collision votes and reopen rule.

        Return type:
        - If allow_partial is False (default): returns a Tensor [T,3] (one-hot actions),
            or an empty [0,3] tensor if unreachable.  (IDENTICAL to current behavior.)
        - If allow_partial is True: returns (Tensor [T,3], meta) where
            meta = {'reached_goal': bool, 'best_dist': float}
            'best_dist' is the Euclidean XY distance of the nearest popped state to the goal.
        """
        import torch
        import torch.nn.functional as F
        import math, heapq

        # ----- helpers -----
        def to_onehot_tensor(seq: list[str]) -> torch.Tensor:
            mapping = {'forward': [1, 0, 0],
                    'right'  : [0, 1, 0],
                    'left'   : [0, 0, 1]}
            return torch.tensor([mapping[a] for a in seq], dtype=torch.float32)

        # >>> CHANGED: tiny helper to pack legacy vs partial-aware returns
        def _pack(seq: list[str], reached: bool, best_dist_val: float):
            tens = to_onehot_tensor(seq)
            if allow_partial:
                return tens, {'reached_goal': bool(reached), 'best_dist': float(best_dist_val)}
            else:
                return tens

        # Monte-Carlo vote for "is there a wall straight ahead *now*?" by sampling *beliefs*.
        # We DO NOT step the environment; we sample latent z variants conditioned on current h.
        def mc_wall_ahead_vote(wm, belief, k: int, ratio: float, debug: bool = False) -> bool:
            z0, h0 = belief
            blocked = 0
            need = max(1, int(math.ceil(k * ratio)))  # votes needed to declare 'blocked'
            with torch.no_grad():
                for i in range(k):
                    z_s = None
                    try:
                        _prior, z_s = wm.rssm.transition_model(h0)
                    except Exception:
                        if debug or verbose:
                            print("   [MC] warning: rssm.transition_model(h) failed; using noisy z")
                        eps = getattr(self, "z_noise_std", 0.05)
                        z_s = z0 + torch.randn_like(z0) * eps

                    if hasattr(self, "is_wall_ahead_now"):
                        if self.is_wall_ahead_now(wm, (z_s, h0), debug=False):
                            blocked += 1

                    # Early exit once the outcome is decided
                    if blocked >= need:
                        if debug or verbose:
                            print(f"   [MC] forward blocked by vote {blocked}/{i+1}")
                        return True
                    if (i + 1 - blocked) > (k - need):
                        break

            if debug or verbose:
                print(f"   [MC] forward free by vote ({blocked}/{k} say blocked)")
            return False

        # ----- config knobs (no signature change) -----
        MC_BLOCK_RATIO = float(getattr(self, "mc_block_ratio", 0.60))  # votes to call it 'blocked'
        H_WEIGHT       = float(getattr(self, "h_weight", 1.3))         # 1.0 == classic A*

        # priority-queue items: (f, g, state, seq, belief)
        pq      = [(H_WEIGHT * self.heuristic(start, goal), 0, start, [], belief_zd)]
        g_score = {start: 0}

        # We still keep a closed set to avoid re-expanding popped nodes *at equal/worse g*.
        # But children are allowed to re-enter if they improve g_score (reopen behavior).
        closed  = set()

        # nearest fallback bookkeeping (used only if allow_partial=True)
        best_dist = float("inf")
        best_seq  = []

        while pq:
            f, g, (x, y, d), seq, belief = heapq.heappop(pq)
            if (x, y, d) in closed:
                continue
            closed.add((x, y, d))

            # update nearest (for allow_partial)
            dist_xy = math.hypot(x - goal.x, y - goal.y)
            if dist_xy < best_dist:
                best_dist, best_seq = dist_xy, seq

            if verbose:
                print(f"[A*] pop {State(x,y,d)}  g={g}  f={f}  seq={seq}")

            # success
            if (x, y, d) == (goal.x, goal.y, goal.d):
                # >>> CHANGED: pack with reached=True if allow_partial else legacy tensor
                return _pack(seq, True, 0.0)

            # budget check (pop boundary)
            if g >= max_actions:
                continue

            # Expand successors (try forward first to exploit promising branches)
            for act in ("forward", "left", "right"):
                # pose after action
                if act == "forward":
                    dx, dy = DIR_VECS[d]
                    nx, ny, nd = x + dx, y + dy, d
                elif act == "left":
                    nx, ny, nd = x, y, (d - 1) % 4
                else:  # right
                    nx, ny, nd = x, y, (d + 1) % 4

                ns = State(nx, ny, nd)

                # g step
                g2 = g + 1
                if g2 > max_actions:
                    continue

                # ---- MC trimming for 'forward' ----
                if act == "forward":
                    if mc_wall_ahead_vote(wm, belief, k=num_rollouts, ratio=MC_BLOCK_RATIO, debug=False):
                        if verbose:
                            print("   prune (MC vote: wall ahead now)  ->", seq + [act])
                        continue

                # belief one step ahead (no-grad)
                with torch.no_grad():
                    z, h = belief
                    onehot = F.one_hot(
                        torch.tensor([wm.idx[act]], device=wm.device),
                        num_classes=wm.action_size
                    ).float()  # (1,3)
                    h_next = wm.rssm.recurrent_model(z, onehot, h)
                    _, z_next = wm.rssm.transition_model(h_next)
                belief_next = (z_next.detach(), h_next.detach())

                new_seq = seq + [act]

                # Heuristic & f (with optional weighting)
                h2 = self.heuristic(ns, goal)
                f2 = g2 + H_WEIGHT * h2

                # ---- reopen rule: push only if we improve g(ns)
                if g2 < g_score.get(ns, float("inf")):
                    g_score[ns] = g2
                    heapq.heappush(pq, (f2, g2, ns, new_seq, belief_next))
                else:
                    if verbose:
                        print(f"   skip {ns} (no g improvement)")

        # failure to reach goal
        if allow_partial and best_seq:
            if verbose:
                print(f"[A*] no path; returning NEAREST (len={len(best_seq)}, dist={best_dist:.2f})")
            # >>> CHANGED: pack with reached=False and best_dist
            return _pack(best_seq, False, best_dist)

        if verbose:
            print("[A*] no path; returning EMPTY")
        # legacy: unreachable → empty tensor
        import torch
        return torch.empty((0, 3), dtype=torch.float32)


    def render_plan(self,wm, belief_zd, actions, include_last=True):
        """
        Return list[Tensor] of decoded RGB frames.
            If include_last=True :  N actions → N+1 frames  (arrival included)
            If include_last=False:  N actions → N   frames  (default behaviour)
        """
        frames = []
        z, h = belief_zd
        with torch.no_grad():
            # 1. current state before the first action
            if include_last:                 # also useful as initial observation
                frame = wm.decoder(z, h).mean.squeeze(0) + 0.5
                frames.append(frame.cpu())

            for act in actions:
                # advance latent one step
                onehot = F.one_hot(torch.tensor([wm.idx[act]], device=wm.device),
                                num_classes=wm.action_size).float()
                h = wm.rssm.recurrent_model(z, onehot, h)
                _, z = wm.rssm.transition_model(h)

                frame = wm.decoder(z, h).mean.squeeze(0) + 0.5
                frames.append(frame.cpu())

        return frames
    def _prepare_frame(self,obs: dict, out_hw=(64, 64)):
        assert "image" in obs, "observation lacks 'image' key"
        img = obs["image"]                                   # H×W×3, uint8 or float
        if img.dtype != np.float32:                          # uint8 → float 0-1
            img = img.astype(np.float32) / 255.0
        if img.shape[:2] != out_hw:                          # resize if needed
            img = cv2.resize(img, out_hw[::-1], interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)                         # to CHW
        img = img - 0.5                                      # centre at 0
        return img                                           # (3,64,64) float32
    def update_belief_from_obs(self,obs_dict,belief_zd, prev_onehot):
            
        frame = self._prepare_frame(obs_dict)       # ★ (3,64,64) float
        return self.wm_update_belief(self.wm, belief_zd, frame, prev_onehot)
    def probe_decoder(self,wm, belief_zd, obs, action_plan, outdir="recon_demo"):
        outdir = Path(outdir); outdir.mkdir(exist_ok=True)
        # ---------- 1. current posterior ------------------------
        rgb = obs["image"].astype(np.float32).transpose(2,0,1)/255.0
        z, d = belief_zd            # posterior of current frame (already updated)
        recon0 = wm.decoder(z, d).mean.squeeze(0).cpu()

        # ---------- 2. imagine K steps (priors) -----------------
        zs, ds, frames = [z], [d], [recon0]
        onehots = torch.stack([onehot(a,wm) for a in action_plan]).unsqueeze(0)

        with torch.no_grad():
            for t in range(len(action_plan)):
                d = wm.rssm.recurrent_model(zs[-1], onehots[:,t], ds[-1])
                _, z = wm.rssm.transition_model(d)   # PRIOR
                zs.append(z);  ds.append(d)
                frames.append( wm.decoder(z, d).mean.squeeze(0).cpu() )

        # ---------- 3. build & save grid ------------------------
        frames = [fr.unsqueeze(0) for fr in frames]          #  → list of 1×3×64×64
        grid   = torch.cat([torch.tensor(rgb).unsqueeze(0)]  # ground-truth 1×3×64×64
                        + frames,
                        dim=0)          
        grid = F.interpolate(grid, size=256,
                                                mode="nearest")  # pixel-art
        fname = outdir / f"probe_{len(list(outdir.glob('probe_*.png'))):03d}.png"
        save_image(grid, fname, nrow=len(frames)+1, normalize=True)
        print("saved", fname)
        
    def save_decoder(self,belief_zd,obs):
                # --- grab one RGB frame from the env -------------------------------
            OUTDIR   = Path("recon_demo")         # ./recon_demo/…
            OUTDIR.mkdir(exist_ok=True)

            #z0,d0= wm.rssm.recurrent_model_input_init(1)                         # posterior (current belief)
            z0,d0=belief_zd
            # ---- choose any action pattern you want the model to fantasise ----
            plan      = ["forward","left", "right", "forward", "forward"]   # length = K
            onehots   = torch.stack([onehot(a, self.wm) for a in plan]).unsqueeze(0)
            zs, ds    = [z0], [d0]

            # ---- latent roll-out (PRIOR predictions!) -------------------------
            for t in range(len(plan)):
                d_next  = self.wm.rssm.recurrent_model(zs[-1], onehots[:, t], ds[-1])
                _, z_next = self.wm.rssm.transition_model(d_next)      # sample from p(zₜ₊₁)
                zs.append(z_next);  ds.append(d_next)

            # ---- decode all latents ------------------------------------------
            with torch.no_grad():
                recons = torch.stack([self.wm.decoder(z, d).mean.squeeze(0).cpu()
                                    for z, d in zip(zs, ds)])   # (K+1,3,64,64)

            # ---- build a pretty  grid  (1×truth  +  K+1×pred) -----------------
            truth = obs["image"].astype(np.float32).transpose(2,0,1)/255.0
            grid  = torch.cat([torch.tensor(truth).unsqueeze(0), recons], dim=0)

            # upscale to 256 px per tile for readability
            grid_big = F.interpolate(grid, size=256,
                                                    mode="bilinear", align_corners=False)

            fname = OUTDIR / f"recon_{len(list(OUTDIR.glob('recon_*.png'))):03d}.png"
            save_image(grid_big, fname, nrow=len(plan)+1, normalize=True)
            print(f"[demo] saved  →  {fname}")

    
    def imagine_decode_embed(self, wm, start_belief, action_seq, *, flatten=True):
        """
        Step Dreamer prior using rssm_step along action_seq, then decode+embed.
        Returns: embedding vector (P,) on CPU (matches dreamer_embed_fn flatten=True).
        """
        belief = start_belief
        for a in action_seq:
            belief = self.rssm_step(wm, belief, a)
        # Reuse your embed path (it already decodes internally in your setup)
        emb = self.dreamer_embed_fn(belief, flatten=flatten)  # (P,) on CPU
        return torch.as_tensor(emb).view(-1).detach().cpu().float()

    def build_enhanced_perception(self, wm, belief_zd, *, combos=None):
        """
        Produce a small set of imagined-view embeddings from the *current* belief.
        combos: list of tuples of actions, default = [('left',), ('right',), ('left','left'), ('right','right')]
        Returns: list[{'seq': tuple[str], 'embed': Tensor(P,)}]
        """
        if combos is None:
            combos = [('left',), ('right',), ('left','left'), ('right','right')]
        out = []
        for seq in combos:
            try:
                emb = self.imagine_decode_embed(wm, belief_zd, seq, flatten=True)
                out.append({'seq': tuple(seq), 'embed': emb})
            except Exception as e:
                # Stay robust; skip bad predictions rather than crashing
                if getattr(self, "_ep_verbose", False):
                    print(f"[EP] warn: failed for seq={seq}: {e}")
                continue
        return out

    def get_recent_enhanced_embeds(self, replay_buffer, K_recent: int, *, skip_last_n: int = 0):
        """
        Collect enhanced predicted embeddings from the last K entries.
        skip_last_n: how many *most recent* entries to skip (e.g., 1 to avoid depth-1 exact match).
        Returns: Tensor (N_pred, P) on CPU, or None if empty.
        """
        preds = []
        # Take last K, optionally skipping the last 'skip_last_n' entries
        items = list(replay_buffer)[-K_recent:] if K_recent > 0 else list(replay_buffer)
        if skip_last_n > 0:
            items = items[:-skip_last_n] if len(items) > skip_last_n else []
        for st in items:
            eps = st.get('enhanced_preds', None)
            if not eps: 
                continue
            for e in eps:
                v = e.get('embed', None)
                if v is not None:
                    preds.append(torch.as_tensor(v).view(-1).detach().cpu().float())
        if len(preds) == 0:
            return None
        return torch.stack(preds, dim=0)
    def hybrid_novelty(
        self,
        cand_vec: torch.Tensor,                # (P,)
        real_bank: Optional[torch.Tensor],     # (N_real, P) or None
        pred_bank: Optional[torch.Tensor],     # (N_pred, P) or None
        *,
        metric: str = "kl",
        mode: str = "weighted_min",            # "weighted_min" | "blend"
        pred_weight: float = 1.25,             # >1.0 reduces pred influence in weighted_min
        blend_beta: float = 0.4                # 0..1, lower => pred contributes less
    ) -> float:
        """
        Returns a single novelty scalar like your old novelty_score.
        - weighted_min: novelty = min(nov_real, pred_weight * nov_pred)   (distance-like scores)
        - blend       : novelty = (1-beta) * nov_real + beta * nov_pred
        If one bank is None, falls back to the other.
        """
        # Helper: reuse existing novelty_score for each bank
        def _nov(bank):
            if bank is None or (hasattr(bank, "numel") and bank.numel() == 0):
                return None
            return float(self.novelty_score(cand_vec, bank, extra_embeds=None, metric=metric))

        nov_r = _nov(real_bank)
        nov_p = _nov(pred_bank)

        # Fallbacks
        if nov_r is None and nov_p is None:
            return 0.0
        if nov_r is None:
            # Only predicted memory available
            if mode == "weighted_min":
                return float(pred_weight * nov_p)
            else:
                return float(nov_p)
        if nov_p is None:
            return float(nov_r)

        # Combine
        if mode == "weighted_min":
            return float(min(nov_r, pred_weight * nov_p))
        else:  # "blend"
            return float((1.0 - blend_beta) * nov_r + blend_beta * nov_p)
    def _dreamer_to_graph_1h(self, a):
        # a: torch or np, 3-d in Dreamer order [L,R,F] → return [F,R,L]
        import numpy as np, torch
        if isinstance(a, torch.Tensor):
            a = a.detach().to("cpu").float().numpy()
        return [float(a[2]), float(a[1]), float(a[0])]

    def update_cog(self, obs_or_img, action_onehot_dreamer, pose_xyz, *,
                place_post=None, place_std=None, std_th=None, agent_lost=False):
        

        if self.cog is None:
            print("[update_cog] cognitive graph not initialized; pass memory_config=... to WMPlanner(...)")
            return

        # 0) action [L,R,F] → [F,R,L] for HE odometry expected by MemoryGraph
        he = self._dreamer_to_graph_1h(action_onehot_dreamer)  # list[3] float, order [F,R,L]

        # 1) pose (we keep it around for debugging / future hooks; MG uses HE odom)
        x, y, th = pose_xyz
        pose_arr = np.asarray([float(x), float(y), float(th)], dtype=np.float32)

        # 2) ExperienceMap context (so first experience doesn’t deref None)
        emap = self.cog.mg.experience_map
        self.emap=emap
        emap.env = self.env
        if getattr(emap, "last_real_pose", None) is None:
            emap.last_real_pose = (float(x), float(y), float(th))
        if getattr(emap, "spawn_pose_real", None) is None:
            emap.spawn_pose_real = (float(x), float(y), float(th))
        # bookkeeping for link creation (optional, safe if these attrs exist)
        if isinstance(action_onehot_dreamer, torch.Tensor):
            act_idx = int(action_onehot_dreamer.argmax().item())
        else:
            act_idx = int(np.argmax(action_onehot_dreamer))
        idx2name = {0: "left", 1: "right", 2: "forward"}
        act_name = idx2name.get(act_idx, "forward")
        emap.last_link_action = act_name
        try:
            emap._recent_prims.append(emap._map_raw_action(act_name))
        except Exception:
            pass

        # 3) image/template
        raw = obs_or_img

        # Lost → do NOT provide an observation (lets MG digest odom, but no new view cell)
        if agent_lost:
            tmpl = None
        else:
            # If we already have a 64-D template, pass it; otherwise, compute it from RGB
            if torch.is_tensor(raw) and raw.ndim == 1 and raw.numel() == 64:
                tmpl = raw.detach().cpu().float()
            elif isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.size == 64:
                import torch as _torch
                tmpl = _torch.from_numpy(raw.astype(np.float32))
            else:
                # Expect HxWx3 (or CHW) → 64-D template
                tmpl = emap.rgb_to_template64(raw, device="cpu")

        # 4) build the observation dict the graph expects
        obs = {
            "image": tmpl,                 # None when lost, else torch.float32(64,)
            "pose":  pose_arr,             # harmless; MG mainly uses HEaction
            "HEaction": he,                # [F,R,L]; HotEncodedActionOdometry will parse this
        }
        # optional: keep raw for debugging
        obs["image_rgb"] = raw

        # 5) update the cognitive graph (internally handles pure-rotation frames)
        self.cog.mg.digest(obs, dt=1, adjust_map=True)

    def get_cog_nodes(self) -> List[Dict[str, Any]]:
        return self.cog.nodes() if self.cog else []
    
    def novelty_astar_tree(
        self,
        wm,
        belief_zd,                  # tuple (z, h) at the *current* frame; each (1, dim)
        start_state,                # State(x,y,d)
        lookahead: int,             # horizon T (e.g., 4)
        replay_buffer,              # deque/list of recent steps
        *,
        K_recent: int = 30,
        metric: str = "kl",         # novelty metric: "kl" | "cos" | "l2"
        lambda_depth: float = 1e-3,
        verbose: bool = False,
        topk: int = 0,              # kept for API compat; unused in this builder
        # viz controls
        viz_tree: bool = False,
        viz_outdir: str = "dbg",
        viz_tag: str = None,
        # weights and hooks
        w_novelty: float = 0.3,
        w_penalty: float = 1.5,     # explicit weight on penalty
        w_p2e: float = 0.0,         # treated as a COST (smaller is better)
        w_graph: float = 0.0,
        gamma: float = 1.0,
        # K-rollout configuration for the uncertainty term (treated as cost)
        p2e_K: int = 4,
        p2e_mode: str = "mc",          # present for API compat; p2e_leaf_value may ignore it
        p2e_S: int = 5,                # if your p2e_leaf_value uses MC internally
        p2e_roll_from: str = "child",  # "child" aligns the probe with edge being evaluated
        p2e_metric: str = "l2",
        # novelty gate
        prune_step_nov_below: float = 0.00,  # set to None to keep all branches
        # graph hook
        memory_graph: Any = None,
        # predicted-memory knobs
        use_pred_memory: bool = True,
        pred_mode: str = "weighted_min",
        pred_weight: float = 1.2,
        pred_blend_beta: float = 0.4,
        skip_pred_for_depth1: bool = True,
    ):
        """
        Build a decision tree over imagined actions with per-node metrics.

        Edge reward at each expansion:
            step_reward = w_novelty * raw_nov
                        - w_penalty * pen
                        - w_p2e     * P2E_Krollout(child_as_cost)
                        + w_graph   * graph_bonus

        Notes:
        - We still apply cheap gates (forward-wall check and optional novelty floor).
        When a child is novelty-pruned, we still create a child node (not enqueued),
        tag it with pruned_reason, and log its metrics.
        - "skip closed" means we've already processed this (pose, depth) once in this pass;
        we avoid duplicate expansions from identical (pose,depth).
        """
        import heapq, time, os
        import numpy as np
        import torch
        F = torch.nn.functional

        # ---------- helpers ----------
        def successor_pose(x, y, d, act):
            if act == "forward":
                dx, dy = DIR_VECS[d]
                return (x + dx, y + dy, d)
            elif act == "left":
                return (x, y, (d - 1) % 4)
            else:  # right
                return (x, y, (d + 1) % 4)

        def step_belief(belief, act: str):
            import torch.nn.functional as F
            z, h = belief  # each (1, dim)
            onehot = F.one_hot(
                torch.tensor([wm.idx[act]], device=wm.device),
                num_classes=wm.action_size
            ).float()
            with torch.no_grad():
                h_next = wm.rssm.recurrent_model(z, onehot, h)
                _, z_next = wm.rssm.transition_model(h_next)
            return (z_next.detach(), h_next.detach())

        # ---------- viz helpers (unchanged) ----------
        if viz_tree:
            import math
            from pathlib import Path
            Path(viz_outdir).mkdir(parents=True, exist_ok=True)

            # We keep these around so old _record() calls don't break, and
            # so we can still fall back to the old collage if PIL is missing.
            _viz_rows = {d: [] for d in range(1, lookahead + 1)}
            _viz_first_hw = None

            # --- Try PIL for pretty tree; if not available, we fall back to collage ---
            try:
                from PIL import Image, ImageDraw, ImageFont
                _pil_ok = True
            except Exception:
                _pil_ok = False

            # no-op-ish annotate (we keep it so existing calls work; used only for fallback)
            def _annotate_tile(img_chw: torch.Tensor, text: str) -> torch.Tensor:
                return img_chw.detach().cpu()

            def _record(depth: int, img_chw: torch.Tensor, label: str):
                # keep minimal work; only used by fallback collage if PIL isn't available
                nonlocal _viz_first_hw
                if depth < 1 or depth > lookahead:
                    return
                img_chw = img_chw.detach().cpu().float()
                if _viz_first_hw is None:
                    _viz_first_hw = (int(img_chw.shape[1]), int(img_chw.shape[2]))
                _viz_rows[depth].append(_annotate_tile(img_chw, label))

            def _finalize_pretty_tree(root_node) -> str:
                """
                Render a tidy layered tree with circular nodes and connecting edges.
                Adds per-node imagined images if available (n.viz_img_chw).
                Also saves a 'plain' version without images for quick scanning.

                Returns path to the image-embedded version.
                """
                if not _pil_ok:
                    # --- Fallback to the previous collage behavior ---
                    if _viz_first_hw is None:
                        return ""
                    from torchvision.utils import make_grid, save_image
                    H, W = _viz_first_hw
                    max_cols = max(len(v) for v in _viz_rows.values()) if _viz_rows else 0
                    if max_cols == 0:
                        return ""
                    gray = torch.ones(3, H, W) * 0.5
                    tiles_all = []
                    for depth in range(1, lookahead + 1):
                        row = _viz_rows.get(depth, [])
                        if len(row) < max_cols:
                            row = row + [gray.clone() for _ in range(max_cols - len(row))]
                        tiles_all.extend(row)
                    grid = make_grid(tiles_all, nrow=max_cols, padding=2)
                    tag = viz_tag if viz_tag else f"{int(time.time()*1000)}"
                    out_path = os.path.join(viz_outdir, f"novastar_tree_{tag}.png")
                    save_image(grid, out_path)
                    return out_path

                import math
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont

                # ---- Collect nodes/edges ----
                nodes, edges = [], []
                def dfs(n):
                    nodes.append(n)
                    for c in n.children:
                        edges.append((n, c))
                        dfs(c)
                dfs(root_node)
                # --- Visible set: only nodes that are NOT pruned and DO have an image tile cached ---
                # --- Visible set: only nodes that are NOT pruned and DO have an image tile cached ---
                eligible = [
                    n for n in nodes
                    if (n.depth >= 0)
                    and (getattr(n, "viz_img_chw", None) is not None)
                    and (not hasattr(n, "pruned_reason"))
                ]
                if not eligible:
                    return ""

                # Use ids for O(1) membership (TreeNode is unhashable)
                eligible_ids = {id(n) for n in eligible}

                # --- Position using ONLY visible children, so hidden/pruned nodes don't affect layout ---
                leaf_counter = 0
                def assign_x_visible(n):
                    nonlocal leaf_counter
                    # Always recurse; only place visible nodes.
                    vis_children = [ch for ch in n.children if id(ch) in eligible_ids]
                    # If this node is not visible, still push into children so visible descendants can be placed.
                    if id(n) not in eligible_ids:
                        for ch in n.children:
                            assign_x_visible(ch)
                        return
                    if not vis_children:
                        setattr(n, "_x", float(leaf_counter))
                        leaf_counter += 1
                    else:
                        for ch in vis_children:
                            assign_x_visible(ch)
                        setattr(n, "_x", sum(getattr(ch, "_x") for ch in vis_children) / len(vis_children))

                assign_x_visible(root_node)

                draw_nodes = eligible
                edges_vis = [(p, c) for (p, c) in edges if (id(p) in eligible_ids and id(c) in eligible_ids)]

                # Bounds from visible nodes only
                xs = [getattr(n, "_x") for n in draw_nodes]
                min_x, max_x = min(xs), max(xs)
                min_depth = min(n.depth for n in draw_nodes)
                max_depth = max(n.depth for n in draw_nodes)


                # ---- Layout constants ----
                HSPACE = 580   # was 200  → more horizontal room between siblings
                VSPACE = 480   # was 140  → more vertical room between levels
                R      = 188    # was 18   → bigger node radius (bigger thumbnails)
                SCALE  = 1    
                MARGIN = 60    # canvas margin
            

                width  = int(MARGIN * 2 + (max_x - min_x + 1) * HSPACE)
                height = int(MARGIN * 2 + (max_depth - min_depth + 1) * VSPACE + 2 * R)

                def render(draw_images: bool) -> Image.Image:
                    W2, H2 = width * SCALE, height * SCALE
                    img = Image.new("RGBA", (W2, H2), (255, 255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("DejaVuSansMono.ttf", size=18 * SCALE*32)
                        font_small = ImageFont.truetype("DejaVuSansMono.ttf", size=12 * SCALE*32)
                    except Exception:
                        font = ImageFont.load_default()
                        font_small = ImageFont.load_default()

                    def npx(xu):  # x unit → px
                        return int((MARGIN + (xu - min_x) * HSPACE) * SCALE)

                    def dpy(depth):  # depth → px (depth=1 is top row)
                        return int((MARGIN + (depth - min_depth) * VSPACE + R) * SCALE)

                    def draw_dashed(p1, p2, dash=10*SCALE, gap=8*SCALE, width=2*SCALE, fill=(150,150,150,255)):
                        x1, y1 = p1; x2, y2 = p2
                        Dx, Dy = x2 - x1, y2 - y1
                        dist = math.hypot(Dx, Dy)
                        if dist == 0:
                            return
                        vx, vy = Dx / dist, Dy / dist
                        t = 0.0
                        while t < dist:
                            seg = min(dash, dist - t)
                            xA = int(x1 + vx * t)
                            yA = int(y1 + vy * t)
                            xB = int(x1 + vx * (t + seg))
                            yB = int(y1 + vy * (t + seg))
                            draw.line([(xA, yA), (xB, yB)], fill=fill, width=width)
                            t += dash + gap

                    # tiny root marker above first layer


                    # ---- Edges first (behind nodes) ----
                    for p, c in edges_vis:
                        x1, y1 = npx(getattr(p, "_x")), dpy(p.depth)
                        x2, y2 = npx(getattr(c, "_x")), dpy(c.depth)
                        draw.line([(x1, y1), (x2, y2)], fill=(0,0,0,255), width=2*SCALE)
                    # ---- Helpers for images ----
                    def chw_to_rgba(img_chw: torch.Tensor) -> Image.Image:
                        t = img_chw
                        if t.dim() == 4:
                            t = t[0]
                        t = t.detach().cpu().float().clamp(0.0, 1.0)
                        if t.shape[0] == 1:
                            t = t.repeat(3, 1, 1)
                        arr = (t.numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                        return Image.fromarray(arr, mode="RGB").convert("RGBA")

                    ACT_ABBR = {"left": "L", "right": "R", "forward": "F"}

                    # ---- Nodes (with optional images) ----
                                        # ---- Nodes (visible only; each must have viz_img_chw) ----
                    for n in draw_nodes:
                        cx, cy = npx(getattr(n, "_x")), dpy(n.depth)
                        r = R * SCALE
                        bb = [cx - r, cy - r, cx + r, cy + r]

                        tile_src = getattr(n, "viz_img_chw", None)
                        if tile_src is not None:
                            try:
                                tile = chw_to_rgba(tile_src)
                                tile = tile.resize((2*r, 2*r), Image.LANCZOS)
                                # circular clip
                                mask = Image.new("L", (2*r, 2*r), 0)
                                mdraw = ImageDraw.Draw(mask)
                                mdraw.ellipse([0, 0, 2*r-1, 2*r-1], fill=255)
                                img.paste(tile, (cx - r, cy - r), mask)
                            except Exception:
                                # extremely unlikely given our eligible filter; draw a white circle fallback
                                draw.ellipse(bb, fill=(255,255,255,255))

                        # crisp outline
                        draw.ellipse(bb, outline=(0,0,0,255), width=2*SCALE)

                        # label inside node
                        ACT_ABBR = {"left": "L", "right": "R", "forward": "F"}
                        act = getattr(n, "step_act", None)
                        if act is None and hasattr(n, "seq") and n.seq:
                            act = n.seq[-1]
                        abbr = ACT_ABBR.get(act, "?")

                        try:
                            tw, th = draw.textbbox((0,0), abbr, font=font)[2:]
                        except Exception:
                            tw, th = draw.textlength(abbr, font=font), 10 * SCALE
                        draw.text(
                            (cx - tw/2, cy - th/2),
                            abbr,
                            fill=(255,255,255,255),
                            font=font,
                            stroke_width=2*SCALE,
                            stroke_fill=(0,0,0,255),
                        )

                        # small metric below (optional; keep if you like)
                        gtext = f"G={getattr(n, 'cum_gain', 0.0):.2f}"
                        try:
                            tw2, th2 = draw.textbbox((0,0), gtext, font=font_small)[2:]
                        except Exception:
                            tw2, th2 = draw.textlength(gtext, font=font_small), 10 * SCALE
                        draw.text((cx - tw2/2, cy + r + 6*SCALE), gtext, fill=(0,0,0,255), font=font_small)


                    
                    return img

                tag = viz_tag if viz_tag else f"{int(time.time()*1000)}"

                # Render with images
                img = render(draw_images=True)
                img_small = img.resize((width, height), Image.LANCZOS).convert("RGB")
                out_img = os.path.join(viz_outdir, f"novastar_tree_{tag}_img.png")
                img_small.save(out_img)


                return out_img

        # ---------- memory banks ----------
        recent_real = self.get_recent_dreamer_embeddings_from_replay(replay_buffer, K_recent)

        recent_pred_all = None
        if use_pred_memory:
            # This expects you to have populated replay entries with 'enhanced_preds'
            # (see build_enhanced_perception / imagine_decode_embed in your codebase).
            recent_pred_all = self.get_recent_enhanced_embeds(replay_buffer, K_recent, skip_last_n=0)

        if verbose:
            print("\n[NOV-A*] ---- init ----")
            print(f"[NOV-A*] metric={metric}  lookahead={lookahead}  K_recent={K_recent}  lambda_depth={lambda_depth}")
            try:
                if hasattr(recent_real, "shape"):
                    print(f"[NOV-A*] recent_real shape={tuple(recent_real.shape)}")
            except Exception as e:
                print(f"[NOV-A*] recent_real introspection failed: {e}")
            if use_pred_memory:
                if recent_pred_all is None:
                    print("[NOV-A*] predicted-memory bank: none (enhanced_preds missing)")
                else:
                    print(f"[NOV-A*] predicted-memory bank: {recent_pred_all.shape[0]} embeddings")
                    print(f"[NOV-A*] pred combine mode={pred_mode}  weight={pred_weight:.2f}  beta={pred_blend_beta:.2f}  skip_depth1={skip_pred_for_depth1}")

        # ---------- PQ item = (key, depth, (x,y,d), seq, belief, gain, local_embeds, local_cells, last_act, node_ref)
        start = start_state
        local_embeds_root = []
        local_cells_root  = {(start.x, start.y)}
        last_act_root     = None

        if not hasattr(self, "TreeNode"):
            raise RuntimeError("WMPlanner.TreeNode not found — ensure dataclass is defined above.")

        _node_id = 0
        root_node = self.TreeNode(
            node_id=_node_id, depth=0, pose=(start.x, start.y, start.d), seq=[],
            belief_zd=(belief_zd[0].detach().cpu(), belief_zd[1].detach().cpu()),
            step_act=None, step_novelty=0.0, step_penalty=0.0, step_p2e=0.0, step_graph=0.0,
            step_reward=0.0, cum_gain=0.0, children=[], is_leaf=False
        )
        if viz_tree:
            try:
                _root_img = self.dreamer_embed_fn(belief_zd, flatten=False)
                setattr(root_node, "viz_img_chw", _root_img.detach().cpu())
            except Exception:
                pass

        pq = [(0.0, 0, (start.x, start.y, start.d), [], belief_zd, 0.0,
            local_embeds_root, local_cells_root, last_act_root, root_node)]
        closed = set()
        import heapq as _heap

        while pq:
            key, g, (x, y, d), seq, belief, gain_so_far, local_embeds, local_cells, last_act, parent_node = _heap.heappop(pq)

            closed_key = (x, y, d, g)
            if closed_key in closed:
                if verbose:
                    print(f"[NOV-A*] skip closed depth={g} pose=({x},{y},{d}) seq={seq} → already expanded this (pose,depth)")
                continue
            closed.add(closed_key)

            if verbose:
                print(f"[NOV-A*] POP depth={g} pose=({x},{y},{d}) G={gain_so_far:.3f} seq={seq} pq={len(pq)}")

            # leaf
            if g >= lookahead:
                parent_node.is_leaf = True
                if parent_node.step_act is not None:
                    parent_node.p2e_leaf_value = self.p2e_leaf_value(
                        wm,
                        (belief[0].detach(), belief[1].detach()),
                        act_name=parent_node.step_act,
                        K=p2e_K,
                        metric=p2e_metric,
                        roll_from=p2e_roll_from
                    )
                    if verbose:
                        print(f"[NOV-A*] leaf p2e({parent_node.step_act})={parent_node.p2e_leaf_value:.3f}")
                else:
                    parent_node.p2e_leaf_value = 0.0
                continue

            for act in ("left", "right", "forward"):
                nx, ny, nd = successor_pose(x, y, d, act)

                # forward collision prune (kept)
                if act == "forward" and self.is_wall_ahead_now(wm, belief, debug=False):
                    if verbose:
                        print(f"   [NOV-A*] PRUNE forward: wall ahead at ({x},{y},{d}) → ({nx},{ny},{nd})")

                    # NEW: compute the "next belief" once, up front
                    with torch.no_grad():
                        _viz_next = step_belief(belief, act)  # (z_next, h_next) on device

                    _node_id += 1
                    pruned_node = self.TreeNode(
                        node_id=_node_id,
                        depth=g+1,
                        pose=(nx, ny, nd),
                        seq=seq + [act],
                        # NEW: store a detached CPU copy so it's safe to keep in the tree
                        belief_zd=(_viz_next[0].detach().cpu(), _viz_next[1].detach().cpu()),
                        step_act=act,
                        step_novelty=float("-inf"),
                        step_penalty=0.0,
                        step_p2e=0.0,
                        step_graph=0.0,
                        step_reward=float("-inf"),
                        cum_gain=float(gain_so_far),
                        children=[],
                        is_leaf=False
                    )
                    setattr(pruned_node, "pruned_reason", "wall_ahead")

                    # NEW: cache a tile now (on CPU) so the renderer doesn't need to re-decode
                    if viz_tree:
                        try:
                            _viz_img = self.dreamer_embed_fn(_viz_next, flatten=False)
                            setattr(pruned_node, "viz_img_chw", _viz_img.detach().cpu())
                        except Exception:
                            pass

                    parent_node.children.append(pruned_node)
                    continue

                # one imagined step
                next_belief = step_belief(belief, act)

                # --- keep this as-is above ---
                img_vec = self.dreamer_embed_fn(next_belief, flatten=True)  # (P,)
                img_chw = None

                if viz_tree:
                    img_chw = self.dreamer_embed_fn(next_belief, flatten=False)
                local_E = torch.stack(local_embeds, dim=0) if local_embeds else None

                # predicted-memory bank selection (unchanged)
                pred_bank = None
                if use_pred_memory and (recent_pred_all is not None):
                    if skip_pred_for_depth1 and g == 0:
                        pred_bank = self.get_recent_enhanced_embeds(replay_buffer, K_recent, skip_last_n=1)
                    else:
                        pred_bank = recent_pred_all

                # ====== NEW: augment the "real" bank with branch-local embeds so hybrid sees them ======
                real_bank_for_hybrid = recent_real
                if (local_E is not None) and (real_bank_for_hybrid is not None):
                    try:
                        rb = torch.as_tensor(real_bank_for_hybrid).float().cpu()
                        real_bank_for_hybrid = torch.cat([rb, local_E.float().cpu()], dim=0)
                    except Exception:
                        # If shape/type mismatch, fall back to original real bank (still safe)
                        real_bank_for_hybrid = recent_real

                # Compute novelty. IMPORTANT: we no longer pass extra_embeds here because we've
                # already concatenated local_E into the real bank to avoid double-counting.
                nov_real = float(self.novelty_score(img_vec, real_bank_for_hybrid, extra_embeds=None, metric=metric))
                if pred_bank is None:
                    raw_nov = nov_real
                else:
                    raw_nov = self.hybrid_novelty(
                        img_vec, real_bank_for_hybrid, pred_bank,
                        metric=metric, mode=pred_mode, pred_weight=pred_weight, blend_beta=pred_blend_beta
                    )
                pen = float(self.pose_recency_penalty(
                    (nx, ny, nd), replay_buffer,
                    local_cells=local_cells,
                    local_last_pose=(x, y, d),
                    last_action=last_act,
                ))

                # K-rollout uncertainty for this child (treated as COST → smaller is better)
                step_p2e = self.p2e_leaf_value(
                    wm,
                    belief if p2e_roll_from == "parent" else next_belief,
                    act_name=act,
                    K=p2e_K,
                    metric=p2e_metric,
                    roll_from=p2e_roll_from
                )

                # graph hook (stub = 0.0 unless you wired it)
                step_graph = self.open_graph_bonus((nx, ny, nd), memory_graph)

                # combine into per-edge reward (explicit penalty + uncertainty as a cost)
                step_reward = (w_novelty * raw_nov) - (w_penalty * pen) - (w_p2e * step_p2e) + (w_graph * step_graph)
                g2 = g + 1
                gain2 = gain_so_far + (gamma ** g) * step_reward

                # novelty floor gate (AFTER computing metrics so we can print/log them)
                if (prune_step_nov_below is not None) and ((raw_nov - pen) < prune_step_nov_below):
                    if verbose:
                        print(f"             PRUNE metric: nov-pen={(raw_nov-pen):.3f} < {prune_step_nov_below:.3f} | "
                            f"raw_nov={raw_nov:.3f} pen={pen:.3f} p2eK={step_p2e:.3f} graph={step_graph:.3f} "
                            f"r={step_reward:.3f} Gwould→{gain2:.3f}")
                    if viz_tree:
                        lab = (f"d={g2} {''.join([s[0].upper() for s in (seq+[act])])} | "
                            f"nov={raw_nov-pen:.2f} p2eK={step_p2e:.2f} gph={step_graph:.2f} PRUNE")
                        _record(g2, img_chw, lab)
                    # attach a PRUNED child with full metrics for visibility
                    _node_id += 1
                    pruned_node = self.TreeNode(
                        node_id=_node_id, depth=g2, pose=(nx, ny, nd), seq=seq + [act],
                        belief_zd=(next_belief[0].detach().cpu(), next_belief[1].detach().cpu()),
                        step_act=act, step_novelty=float(raw_nov - pen), step_penalty=float(pen),
                        step_p2e=float(step_p2e), step_graph=float(step_graph),
                        step_reward=float(step_reward), cum_gain=float(gain_so_far), children=[], is_leaf=False
                    )
                    setattr(pruned_node, "pruned_reason", "novelty_floor")
                    parent_node.children.append(pruned_node)
                    if viz_tree and (img_chw is not None):
                        setattr(pruned_node, "viz_img_chw", img_chw.detach().cpu())
                    continue

                if verbose:
                    print(f"   [NOV-A*] a={act:<6} → ({nx},{ny},{nd})  "
                        f"raw_nov={raw_nov:.3f}  pen={pen:.3f}  p2eK={step_p2e:.3f}  graph={step_graph:.3f}  "
                        f"r={step_reward:.3f}  G→{gain2:.3f}")

                if viz_tree:
                    lab = (f"d={g2} {''.join([s[0].upper() for s in (seq+[act])])} | "
                        f"nov={raw_nov-pen:.2f} p2eK={step_p2e:.2f} gph={step_graph:.2f} r={step_reward:.2f} G={gain2:.2f}")
                    _record(g2, img_chw, lab)

                # carry local state for novelty
                new_local_embeds = local_embeds + [torch.as_tensor(img_vec).float().cpu()]
                new_local_cells = set(local_cells); new_local_cells.add((nx, ny))

                # A*-like key controls expansion order (not pruning)
                new_key = -(gain2) + lambda_depth * g2

                # build child node
                _node_id += 1
                child_node = self.TreeNode(
                    node_id=_node_id, depth=g2, pose=(nx, ny, nd), seq=seq + [act],
                    belief_zd=(next_belief[0].detach().cpu(), next_belief[1].detach().cpu()),
                    step_act=act, step_novelty=float(raw_nov - pen), step_penalty=float(pen),
                    step_p2e=float(step_p2e), step_graph=float(step_graph),
                    step_reward=float(step_reward), cum_gain=float(gain2), children=[], is_leaf=False
                )
                parent_node.children.append(child_node)
                if viz_tree and (img_chw is not None):
                    setattr(child_node, "viz_img_chw", img_chw.detach().cpu())

                heapq.heappush(
                    pq,
                    (new_key, g2, (nx, ny, nd), seq + [act], next_belief, gain2,
                    new_local_embeds, new_local_cells, act, child_node)
                )

        if viz_tree:
            path = _finalize_pretty_tree(root_node)
            self._last_novastar_tree_path = path
            if verbose and path:
                print(f"[NOV-A*] saved decision tree → {path}")

        root_node.is_leaf = (lookahead == 0)
        return root_node


    # ─────────────────────────────────────────────────────────────────────
    # SHIM: keep the old API, but drive the new tree builder under the hood
    # ─────────────────────────────────────────────────────────────────────
    def novelty_astar_plan(
        self, wm, belief_zd, start_state, lookahead, replay_buffer, *,
        K_recent: int = 30, metric: str = "kl", lambda_depth: float = 1e-3,
        verbose: bool = False, topk: int = 0,
        viz_tree: bool = False, viz_outdir: str = "dbg", viz_tag: str = None,
        # weights
        w_p2e: float = 0.0, w_novelty: float = 1.0, w_graph: float = 0.0,
        gamma: float = 1.0,
        # K-rollout config
        p2e_K: int = 4, p2e_mode: str = "mc", p2e_S: int = 5,
        p2e_roll_from: str = "child", p2e_metric: str = "l2",
        memory_graph: Any = None,
    ):
        """
        Wrapper to keep old API ((T,3) one-hot). We first build the *full tree*,
        then pick an action by scanning the root’s children for max cum_gain.
        """
        def to_onehot(seq: List[str]) -> torch.Tensor:
            mapping = {'forward': [1, 0, 0], 'right': [0, 1, 0], 'left': [0, 0, 1]}
            return torch.tensor([mapping[a] for a in seq], dtype=torch.float32)
        self._last_memory_graph = memory_graph
        tree = self.novelty_astar_tree(
            wm, belief_zd, start_state, lookahead, replay_buffer,
            K_recent=K_recent, metric=metric, lambda_depth=lambda_depth,
            verbose=verbose, topk=0, viz_tree=viz_tree, viz_outdir=viz_outdir, viz_tag=viz_tag,
            w_p2e=w_p2e, w_novelty=w_novelty, w_graph=w_graph, gamma=gamma,
            p2e_K=p2e_K, p2e_roll_from=p2e_roll_from, p2e_metric=p2e_metric,
            memory_graph=memory_graph,
        )
        self._last_decision_tree = tree

        best_action, action_scores, top_paths = self.first_action_from_topk_paths(
            tree,
            k=10,                       # change if you want a different K
            include_pruned=True,
            weight="linear",            # "linear" | "harmonic" | "exp"
            alpha=0.85,                 # only used for weight="exp"
            require_min_depth=1,
            verbose=verbose,
        )


        if best_action is None:
            return to_onehot([])
        return to_onehot([best_action])
        # ─────────────────────────────────────────────────────────────────────
    # Dynamic Programming backup over the decision tree
    # ─────────────────────────────────────────────────────────────────────
    def _p2e_leaf_bonus(self, node, *, coef: float, lower_is_better: bool) -> float:
        """Scalar terminal bonus from p2e_leaf_value (None -> 0)."""
        v = 0.0 if node.p2e_leaf_value is None else float(node.p2e_leaf_value)
        # You asked: "less p2e_leaf_value is BETTER (more uncertainty)" → invert.
        signed = -v if lower_is_better else +v
        return coef * signed

    
    
    def _graph_nodes_for_ranking(self, *, dbg=False):
        """
        Return (nodes_xy:list[(x,y)], weights:list[float]) using EXACTLY the same
        sources your visualizer uses:
        1) get_memory_map_data(dbg=False): uses self.cog.mg + experience_map
        2) fallback: self.cog.mg.view_cells.cells[*].exps and mg.experience_map.exps
        """
        def _norm_pairs(pairs):
            xy = []
            for p in pairs:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0], p[1]
                    if x is None or y is None:
                        continue
                    xy.append((float(x), float(y)))
            return xy

        # --- 1) Preferred: the exact data structure your viz builds ---
        try:
            if hasattr(self, "get_memory_map_data") and callable(self.get_memory_map_data):
                data = self.get_memory_map_data(dbg=False)
                exps = data.get("exps_GP", []) or []
                decays = data.get("exps_decay", []) or []
                xy = _norm_pairs(exps)
                if xy:
                    use_decay = bool(getattr(self, "rank_graph_use_decay_weights", False))
                    if use_decay and decays and len(decays) == len(xy):
                        w = [float(d) for d in decays]
                    else:
                        w = [1.0] * len(xy)
                    if dbg:
                        print(f"[RANK:cog] nodes from get_memory_map_data: {len(xy)} (weights={'decay' if use_decay else '1s'})")
                    return xy, w
        except Exception as e:
            if dbg:
                print(f"[RANK:cog] get_memory_map_data failed: {e}")

        # --- 2) Fallback: mirror get_memory_map_data internals directly ---
        try:
            cog = getattr(self, "cog", None)
            if cog is None:
                if dbg: print("[RANK:cog] self.cog is None")
                return [], []
            mg = getattr(cog, "mg", None)
            if mg is None:
                if dbg: print("[RANK:cog] self.cog.mg is None")
                return [], []

            xy, w = [], []
            use_decay = bool(getattr(self, "rank_graph_use_decay_weights", False))

            # Experiences via view_cells (nodes + decay)
            try:
                vcc = getattr(mg, "view_cells", None)
                cells = getattr(vcc, "cells", []) if vcc is not None else []
                for vc in cells:
                    dec = float(getattr(vc, "decay", 1.0))
                    for exp in getattr(vc, "exps", []):
                        x = getattr(exp, "x_m", None)
                        y = getattr(exp, "y_m", None)
                        if x is None or y is None:
                            continue
                        xy.append((float(x), float(y)))
                        w.append(dec if use_decay else 1.0)
            except Exception:
                pass

            # Experiences via experience_map (nodes)
            try:
                emap = getattr(mg, "experience_map", None)
                if emap is not None:
                    for exp in getattr(emap, "exps", []):
                        x = getattr(exp, "x_m", None)
                        y = getattr(exp, "y_m", None)
                        if x is None or y is None:
                            continue
                        xy.append((float(x), float(y)))
                        w.append(1.0)
            except Exception:
                pass

            # Deduplicate (tolerant to float fuzz)
            if xy:
                seen, xy_d, w_d = set(), [], []
                for (x, y), ww in zip(xy, w):
                    key = (round(x, 4), round(y, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    xy_d.append((x, y)); w_d.append(float(ww))
                if dbg:
                    print(f"[RANK:cog] nodes from self.cog.mg: {len(xy_d)}")
                return xy_d, w_d

            if dbg:
                print("[RANK:cog] no nodes found in cog.mg")
            return [], []
        except Exception as e:
            if dbg:
                print(f"[RANK:cog] fallback via cog.mg failed: {e}")
            return [], []


    def first_action_from_topk_paths(
        self,
        root,
        *,
        k: int = 10,
        include_pruned: bool = True,
        weight: str = "linear",     # "linear" | "harmonic" | "exp"
        alpha: float = 0.85,        # only used when weight="exp"
        require_min_depth: int = 1, # ignore empty sequences (no first action)
        verbose: bool = False,
    ):
        """
        Rank root→terminal paths by a NEW score that ignores the builder's 'gain':
            score(path) = new_gain_from_steps
                        + B * dist_from_start
                        + C * dist_from_graph_nodes   (now computed against ALL graph nodes)

        - new_gain_from_steps uses only per-step metrics stored in TreeNode.
        - dist_from_graph_nodes uses a robust aggregator over ALL graph nodes (default: RBF separation).

        Everything else (top-k, rank-weighted first-action voting) is unchanged.
        Returns:
            best_action : str   in {"left","right","forward"}
            action_scores : dict[str, float]
            top_paths : list[dict] enriched with 'score' and 'score_terms'
        """

        # =========================
        # Tunables (override on planner after init)
        # =========================
        Wp2e        = getattr(self, "rank_Wp2e", 1.0)       # weight for sibling-normalized p2e in step value
        gammaR      = getattr(self, "rank_gamma", 1.0)      # per-step discount in ranking sum (1.0 = none)
        Bdist       = getattr(self, "rank_B", 1.0 )         # weight for distance from start
        Cgraph      = getattr(self, "rank_C", 1.5)         # weight for graph separation term
        dist_metric = getattr(self, "rank_dist_metric", "manhattan")  # or "euclidean"

        # Graph aggregation controls
        graph_mode  = getattr(self, "rank_graph_mode", "rbf_softmin")  # "rbf_sep" | "knn_mean" | "power_mean" | "min" | "mean"
        graph_sigma = float(getattr(self, "rank_graph_sigma", 8.5))# for rbf_sep: larger = broader influence (tiles)
        graph_k     = int(getattr(self, "rank_graph_k", 3))        # for knn_mean
        graph_p     = float(getattr(self, "rank_graph_p", 2.0))    # for power_mean
    
        # Optional:
        self.rank_dist_metric = "manhattan"  # or "euclidean"
        # =========================
        # Helpers
        # =========================
        def _dist_xy(pose, xy):
            x0, y0, _ = pose
            x1, y1    = xy
            dx = abs(float(x1) - float(x0))
            dy = abs(float(y1) - float(y0))
            if dist_metric == "euclidean":
                return (dx*dx + dy*dy) ** 0.5
            return dx + dy  # manhattan


        def _path_nodes_by_seq(root_node, seq):
            """Return list of nodes along seq (excluding root), or [] if not found."""
            nodes = []
            cur = root_node
            prefix = []
            for a in seq:
                prefix.append(a)
                nxt = None
                # prefer exact-seq match
                for ch in cur.children:
                    if getattr(ch, "seq", None) == prefix:
                        nxt = ch
                        break
                if nxt is None:
                    # fallback: first child with action
                    for ch in cur.children:
                        if getattr(ch, "step_act", None) == a:
                            nxt = ch
                            break
                if nxt is None:
                    return []
                nodes.append(nxt)
                cur = nxt
            return nodes

        def _sibling_p2e_norm_inverted(child_node, parent_node):
            """Normalize this child's step_p2e among siblings and invert ⇒ smaller p2e → larger bonus ∈ [0,1]."""
            try:
                sibs = [c for c in getattr(parent_node, "children", []) if hasattr(c, "step_p2e")]
                vals = [float(c.step_p2e) for c in sibs if (c.step_p2e is not None) and (float(c.step_p2e) >= 0.0)]
                v = float(child_node.step_p2e)
                if not vals or len(vals) == 1:
                    return 0.5
                vmin, vmax = min(vals), max(vals)
                if vmax <= vmin:
                    return 0.5
                return (vmax - v) / (vmax - vmin + 1e-8)
            except Exception:
                return 0.5

        # ---- graph nodes collection (robust to several backends) ---


        def _graph_separation(end_pose, nodes_xy, nodes_w):
                """
                Aggregates distances to ALL nodes into a single scalar.
                Modes:
                - rbf_sep (default): sep = 1 - (Σ w_i * exp(-d_i/σ)) / (Σ w_i)
                - knn_mean:          mean of k nearest distances
                - power_mean:        (Σ w_i * d_i^p / Σ w_i)^(1/p)
                - min / mean:        obvious
                Returns a value where LARGER means "farther from graph overall".
                """
                import math  
                if not nodes_xy:
                    return 0.0

                # distances
                D = [_dist_xy(end_pose, xy) for xy in nodes_xy]

                if graph_mode == "rbf_sep":
                    import math
                    sigma = max(1e-6, graph_sigma)
                    wsum = sum(nodes_w) if nodes_w else float(len(D))
                    dens = 0.0
                    for d, w in zip(D, nodes_w if nodes_w else [1.0]*len(D)):
                        dens += w * math.exp(-d / sigma)
                    dens /= max(wsum, 1e-6)
                    sep = 1.0 - dens
                    return float(sep)
                elif graph_mode == "rbf_softmin":
                    sigma = max(1e-6, graph_sigma)
                    W = nodes_w if nodes_w else [1.0] * len(D)

                    # Stable soft-min: shift by the minimum distance
                    d0 = min(D)

                    # Normalize by total weight to remove cluster/count bias
                    wsum = max(sum(W), 1e-12)
                    s = 0.0
                    for d, w in zip(D, W):
                        s += w * math.exp(-(d - d0) / sigma)
                    s /= wsum  # s ∈ (0, 1], independent of number of nodes

                    # Soft-min, no zero clamp (keeps small but informative values)
                    d_softmin = d0 - sigma * math.log(max(s, 1e-12))
                    return float(d_softmin)

                elif graph_mode == "knn_mean":
                    k = max(1, min(graph_k, len(D)))
                    return float(sum(sorted(D)[:k]) / k)

                elif graph_mode == "power_mean":
                    p = float(graph_p)
                    if abs(p) < 1e-9:
                        # p ~ 0 → geometric mean; avoid log for simplicity: fall back to mean
                        return float(sum(D) / len(D))
                    wsum = sum(nodes_w) if nodes_w else float(len(D))
                    num = 0.0
                    for d, w in zip(D, nodes_w if nodes_w else [1.0]*len(D)):
                        num += w * (d ** p)
                    return float((num / max(wsum, 1e-6)) ** (1.0 / p))

                elif graph_mode == "min":
                    return float(min(D))

                else:  # "mean"
                    return float(sum(D) / len(D))

        # =========================
        # 1) Enumerate all paths from the tree (we ignore its stored 'gain')
        # =========================
        ranked_all = self.rank_paths(root, include_pruned=include_pruned)

        # Precollect graph nodes once
        graph_nodes_xy, graph_nodes_w = self._graph_nodes_for_ranking(dbg=verbose)
        if verbose:
            print(f"[DEBUG cog] nodes={len(graph_nodes_xy)}")
        # =========================
        # 2) Compute NEW score per path
        # =========================
        start_pose = getattr(root, "pose", None) or (None, None, None)
        enriched = []
        for rec in ranked_all:
            seq   = rec.get("seq", [])
            depth = int(rec.get("depth", len(seq) or 0))
            if depth < require_min_depth:
                continue

            # Traverse to get actual nodes (to access per-step metrics w/ sibling context)
            step_nodes = _path_nodes_by_seq(root, seq)
            if not step_nodes:
                continue

            # Accumulate per-step contributions
            new_gain = 0.0
            parent = root
            for t, node in enumerate(step_nodes, start=1):
                step_pen = float(getattr(node, "step_penalty", 0.0))
                step_nov_minus_pen = float(getattr(node, "step_novelty", 0.0))
                raw_nov = step_nov_minus_pen + step_pen  # since step_novelty = raw_nov - pen

                p2e_bonus = _sibling_p2e_norm_inverted(node, parent)  # ∈ [0,1]
                step_value = (raw_nov - step_pen) + (Wp2e * p2e_bonus)

                new_gain += (gammaR ** (t - 1)) * step_value
                parent = node

            # Distance terms
            dist_start = 0.0
            end_pose = rec.get("end_pose", getattr(step_nodes[-1], "pose", None))
            if start_pose[0] is not None and end_pose is not None:
                dist_start = _dist_xy(start_pose, (end_pose[0], end_pose[1]))

            dist_graph_all = 0.0
            if graph_nodes_xy:
                use_path_min = bool(getattr(self, "rank_graph_use_path_min", True))
                if use_path_min:
                    poses = [getattr(n, "pose", None) for n in step_nodes if getattr(n, "pose", None) is not None]
                    # Ensure end pose considered (in case last node lacks pose)
                    if end_pose is not None and (not poses or poses[-1] != end_pose):
                        poses.append(end_pose)
                    vals = [_graph_separation(p, graph_nodes_xy, graph_nodes_w) for p in poses if p is not None]
                    dist_graph_all = min(vals) if vals else 0.0
                else:
                    dist_graph_all = _graph_separation(end_pose, graph_nodes_xy, graph_nodes_w)

                print("[DEBUG cog2]", dist_graph_all)

            score = float(new_gain + Bdist * dist_start + Cgraph * dist_graph_all)

            rec2 = dict(rec)
            rec2["score"] = score
            rec2["score_terms"] = {
                "new_gain": float(new_gain),
                "dist_from_start": float(dist_start),
                "dist_from_graph_nodes": float(dist_graph_all),
                "B": float(Bdist),
                "C": float(Cgraph),
                "Wp2e": float(Wp2e),
                "gamma_rank": float(gammaR),
                "graph_mode": str(graph_mode),
            }
            enriched.append(rec2)

        # =========================
        # 3) Sort by NEW score (DESC) and take top-k
        # =========================
        ranked_desc = sorted(enriched, key=lambda r: r["score"], reverse=True)
        print(ranked_desc)
        top_paths = ranked_desc[:k]

        # =========================
        # 4) Rank-weighted FIRST-action voting (unchanged)
        # =========================
        actions = ("left", "right", "forward")
        scores = {a: 0.0 for a in actions}
        contrib = {a: [] for a in actions}

        def rank_weight(i: int) -> float:
            if weight == "harmonic":
                return 1.0 / (i + 1)
            if weight == "exp":
                return float(alpha ** i)
            return float(max(k - i, 1))  # linear

        for i, rec in enumerate(top_paths):
            seq = rec.get("seq", [])
            if len(seq) < require_min_depth:
                continue
            a0 = seq[0]
            if a0 not in scores:
                continue
            w = rank_weight(i)
            scores[a0] += w
            contrib[a0].append((i + 1, w, rec.get("score", 0.0), list(seq)))

        def tiebreak(kv):
            order = {"forward": 2, "left": 1, "right": 0}
            return (kv[1], order.get(kv[0], -1))

        best_action = max(scores.items(), key=tiebreak)[0]

        # =========================
        # 5) Verbose debug
        # =========================
        if verbose:
            print("\n[TOP-K FIRST-ACTION VOTING]")
            print(f"  k={k} weight={weight} alpha={alpha} include_pruned={include_pruned}")
            for i, rec in enumerate(ranked_desc, 1):
                a0 = rec["seq"][0] if rec["seq"] else "∅"
                s_terms = rec.get("score_terms", {})
                print(f"  #{i:02d} S={rec['score']:.3f}  first={a0:<7} depth={rec.get('depth')} seq={rec.get('seq')} "
                    f"pruned={rec.get('pruned_reason')}")
                print(f"        └ new_gain={s_terms.get('new_gain', 0.0):.3f}  "
                    f"B*dist_start={s_terms.get('B',0.0)*s_terms.get('dist_from_start',0.0):.3f}  "
                    f"C*graph={s_terms.get('C',0.0)*s_terms.get('dist_from_graph_nodes',0.0):.3f}  "
                    f"[graph_mode={s_terms.get('graph_mode','?')}]")

            print("  Action scores:")
            for a in actions:
                print(f"    {a:<7} → {scores[a]:.3f}  via {len(contrib[a])} hits")
                for rank, w, S, s in contrib[a]:
                    print(f"       - rank#{rank:02d} w={w:.3f} S={S:.3f} seq={s}")

        if verbose:
            print("\n[TOP-K FIRST-ACTION VOTING]")
            print(f"  k={k} weight={weight} alpha={alpha} include_pruned={include_pruned}")
            for i, rec in enumerate(top_paths, 1):
                a0 = rec["seq"][0] if rec["seq"] else "∅"
                s_terms = rec.get("score_terms", {})
                print(f"  #{i:02d} S={rec['score']:.3f}  first={a0:<7} depth={rec.get('depth')} seq={rec.get('seq')} "
                    f"pruned={rec.get('pruned_reason')}")
                print(f"        └ new_gain={s_terms.get('new_gain', 0.0):.3f}  "
                    f"B*dist_start={s_terms.get('B',0.0)*s_terms.get('dist_from_start',0.0):.3f}  "
                    f"C*graph={s_terms.get('C',0.0)*s_terms.get('dist_from_graph_nodes',0.0):.3f}  "
                    f"[graph_mode={s_terms.get('graph_mode','?')}]")

            print("  Action scores:")
            for a in actions:
                print(f"    {a:<7} → {scores[a]:.3f}  via {len(contrib[a])} hits")
                for rank, w, S, s in contrib[a]:
                    print(f"       - rank#{rank:02d} w={w:.3f} S={S:.3f} seq={s}")

        self._last_top_paths = top_paths
        self._last_action_scores = scores
        return best_action, scores, top_paths

    def rank_paths(self, root, *, include_pruned: bool = True):
        """
        Enumerate and rank all root→terminal paths by cumulative gain (desc).
        'Terminal' = (is_leaf==True) OR (pruned child) OR (no children).
        Returns: list of dicts {gain, seq, depth, end_pose, leaf_p2e, pruned_reason}
        """
        out = []

        def is_terminal(node):
            # terminal if explicitly leaf OR all children are pruned OR no children
            if getattr(node, "is_leaf", False):
                return True
            if not node.children:
                return True
            if all(getattr(c, "pruned_reason", None) is not None for c in node.children):
                return True
            return False

        def dfs(node):
            if is_terminal(node):
                out.append({
                    "gain": float(node.cum_gain),
                    "seq": list(node.seq),
                    "depth": int(node.depth),
                    "end_pose": tuple(node.pose),
                    "leaf_p2e": float(node.p2e_leaf_value) if getattr(node, "p2e_leaf_value", None) is not None else None,
                    "pruned_reason": getattr(node, "pruned_reason", None),
                })
                return
            for c in node.children:
                if (getattr(c, "pruned_reason", None) is not None) and not include_pruned:
                    continue
                dfs(c)

        dfs(root)
        out.sort(key=lambda d: d["gain"], reverse=False)
        
        return out


    
    def dreamer_embed_fn(self, belief, *, flatten: bool = True, scale01: bool = True):
        """
        Decode a Dreamer belief (z,h) to an image and (optionally) flatten it for similarity.
        Returns a CPU tensor:
        - if flatten=True: (C*H*W,) for cosine/L2 comparisons
        - else: (C,H,W)
        """
        import torch

        if belief is None:
            return torch.empty(0)

        z, h = belief  # (1,Z), (1,H)
        with torch.no_grad():
            # Use the simple, native decoder call pattern you’re already using elsewhere.
            # Expect outputs in Dreamer scale ~[-0.5, 0.5]; shift to [0,1] if requested.
            dec = None
            if hasattr(self.wm, "decoder"):
                dec = self.wm.decoder(z, h)
            elif hasattr(self.wm, "rssm") and hasattr(self.wm.rssm, "decoder"):
                dec = self.wm.rssm.decoder(z, h)
            else:
                raise AttributeError("No decoder found on wm (expected wm.decoder or wm.rssm.decoder)")

            # Distribution or tensor
            x = getattr(dec, "mean", dec)
            x = x.squeeze(0)  # (C,H,W)

            if scale01:
                x = (x + 0.5).clamp(0.0, 1.0)  # to [0,1] for image cosine/L2 stability

            x = x.detach().to("cpu")
            return x.reshape(-1) if flatten else x

    def novelty_score(
        self,
        z_pred,
        recent_embeds,
        *,
        extra_embeds=None,      # OPTIONAL: (G,P) tensor of path-local flattened images
        metric: str = "kl",
        age_tau: float = 8.0,   # NEW: larger = slower decay ⇒ older items still matter; smaller = emphasize recency
    ) -> torch.Tensor:
        """
        Compute novelty of z_pred (flattened image) vs union of:
        - recent_embeds: (K,P) from replay (assumed chronological oldest→newest)
        - extra_embeds:  (G,P) from current path (assumed earliest→latest)

        Recency weighting:
        - Each comparison gets a weight w(age) = exp(-age / age_tau), where age=0 is the MOST RECENT item.
        - Recent items (age≈0) get w≈1 → they dominate; old items get w→0 → they barely penalize novelty.
        - This prevents pruning just because something looked similar 20–40 steps ago.

        Metrics:
        - 'cos' (recommended for images): novelty = 1 - max_i( w_i * cos_sim_i )
        - 'l2' : novelty = min_i( ||x_i - z|| / max(w_i, eps) )
        - 'kl' : novelty = min_i( 0.5 * ||x_i - z||^2 / max(w_i, eps) )

        Returns: scalar 0-D torch.Tensor (CPU).
        """
        import torch
        import torch.nn.functional as F

        # ---- Collect comparison sets ----
        parts = []
        weights = []

        # Helper to append a block and its recency weights from oldest→newest rows
        def _append_block(block):
            B = torch.as_tensor(block).float()
            if B.ndim == 1:
                B = B.unsqueeze(0)
            if B.numel() == 0:
                return None
            # ages: newest row has age=0
            n = B.shape[0]
            ages = torch.arange(n - 1, -1, -1, dtype=torch.float32)  # [n-1, ..., 1, 0]
            w = torch.exp(-ages / float(age_tau))                    # exp decay; newest→1.0
            parts.append(B)
            weights.append(w)
            return B

        if recent_embeds is not None:
            _append_block(recent_embeds)
        if extra_embeds is not None:
            _append_block(extra_embeds)

        if not parts:
            return torch.tensor(1.0)

        # Concatenate comparisons and their weights
        Eall = torch.cat(parts, dim=0)                 # (M, P)
        Wall = torch.cat(weights, dim=0)               # (M,)
        z = torch.as_tensor(z_pred).float().view(-1)   # (P,)
        eps = 1e-6

        with torch.no_grad():
            if metric == "cos":
                # Cosine similarity with recency weighting on the similarities
                z_n = F.normalize(z, dim=0)
                E_n = F.normalize(Eall, dim=1)
                sims = E_n @ z_n                        # (M,)
                wsims = Wall * sims                     # weight recent matches more
                novelty = 1.0 - torch.clamp(wsims.max(), -1.0, 1.0)

            elif metric == "l2":
                # Make recent near-neighbors *more influential* by dividing distance by weight
                dists = torch.linalg.norm(Eall - z, dim=1)          # (M,)
                eff = dists / torch.clamp(Wall, min=eps)
                novelty = eff.min()

            elif metric == "kl":
                diffs = Eall - z
                sq = 0.5 * torch.sum(diffs * diffs, dim=1)          # (M,)
                eff = sq / torch.clamp(Wall, min=eps)
                novelty = eff.min()

            else:
                raise ValueError(f"Unknown metric '{metric}'")

            if torch.isnan(novelty):
                novelty = torch.tensor(0.0)

            return novelty

    def get_recent_dreamer_embeddings_from_replay(self, replay_buffer, K_recent: int):
        """
        Return up to K_recent *decoded images* from the replay buffer, flattened to 1D.
        Shape: (N<=K_recent, C*H*W) on CPU.
        Tries keys in each state:
        - 'dreamer_decoded' (preferred, (C,H,W) or (1,C,H,W))
        - else decodes from 'belief_zd' or 'belief' if present
        """
        import torch

        if not replay_buffer or K_recent <= 0:
            return torch.empty(0, dtype=torch.float32)

        outs = []
        count = 0
        # newest → oldest
        for st in reversed(replay_buffer):
            x = st.get("dreamer_decoded", None)

            if x is None:
                belief = st.get("belief_zd", None) or st.get("belief", None)
                if belief is not None:
                    try:
                        x = self.dreamer_embed_fn(belief, flatten=False, scale01=True)  # (C,H,W)
                    except Exception:
                        x = None

            if x is None:
                continue

            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 4 and x.shape[0] == 1:  # (1,C,H,W) → (C,H,W)
                x = x[0]
            if x.ndim == 3 and x.shape[0] in (1, 3):
                pass
            elif x.ndim == 3 and x.shape[-1] in (1, 3):  # HWC → CHW
                x = x.permute(2, 0, 1).contiguous()
            elif x.ndim == 2:
                x = x.unsqueeze(0)  # make (1,H,W)

            outs.append(x.reshape(-1).cpu())
            count += 1
            if count >= K_recent:
                break

        if not outs:
            return torch.empty(0, dtype=torch.float32)

        # chronological (oldest→newest) for sanity; order doesn't affect max/nn ops
        return torch.stack(list(reversed(outs)), dim=0)


    def pose_recency_penalty(
        self,
        candidate_pose,
        replay_buffer,
        *,
        local_cells=None,               # set of (x,y) from the CURRENT PATH (may be empty set)
        local_last_pose=None,           # (x,y,d) of parent node to detect turn-in-place
        window: int = 40,               # how many recent replay states to consider
        spin_small: float = 0.05,       # legacy small spin penalty (used if last_action=None or first_turn_free=False)
        turn_again_penalty: float = 0.35,  # stronger penalty for turn-after-turn
        revisit_large: float = 0.35,    # big penalty for returning to a known cell
        tau: float = 6.0,               # recency decay for replay-based cell revisits
        last_action: str | None = None, # previous action in the branch ("left"|"right"|"forward"|None)
        first_turn_free: bool = True,   # if True, first turn after a forward is unpenalized
    ) -> float:
        """
        Adaptive penalties:

        (A) TURN-IN-PLACE (same cell, orientation changes):
            - If last_action was 'forward' (or None) and first_turn_free=True → 0 penalty.
            - If last_action was a turn ('left' or 'right') → apply `turn_again_penalty`.
            - If last_action is None and you want legacy behavior → set first_turn_free=False to use `spin_small`.

        (B) MOVE INTO A PREVIOUSLY SEEN CELL:
            - If the next cell is already in THIS path → immediate `revisit_large`.
            - Else if that cell appears in recent replay → `revisit_large * exp(-age/tau)` (age=0 is most recent).

        IMPORTANT FIX:
            - The replay-based revisit penalty is applied ONLY if the step MOVES to a new cell.
            Turning in place (same (x,y)) will NOT trigger the heavy replay penalty.
        """
        import math
        from itertools import islice

        if candidate_pose is None:
            return 0.0

        x, y, d = map(int, candidate_pose)
        pen = 0.0

        # ----- (A) adaptive turn-in-place handling -----
        moved_to_new_cell = True  # default; will correct below if we know parent pose
        if local_last_pose is not None:
            lx, ly, ld = map(int, local_last_pose)
            same_cell = (x, y) == (lx, ly)
            moved_to_new_cell = not same_cell
            turned = same_cell and (d != ld)

            if turned:
                if last_action in ("left", "right"):
                    # consecutive turning → stronger penalty
                    pen += float(turn_again_penalty)
                else:
                    # previous was forward (or unknown)
                    if first_turn_free:
                        pen += 0.0
                    else:
                        pen += float(spin_small)
        else:
            # No parent pose provided (e.g., very first call): if we *know* the last action was a turn,
            # treat this as a turn-in-place for the purpose of replay penalty gating.
            if last_action in ("left", "right"):
                moved_to_new_cell = False  # prevents heavy replay penalty on first turn

        # ----- (B1) local-path revisit penalty (immediate) -----
        if local_cells is not None and local_last_pose is not None:
            lx, ly, _ = map(int, local_last_pose)
            if (x, y) != (lx, ly) and (x, y) in local_cells:
                pen = max(pen, float(revisit_large))

        # ----- (B2) replay-based revisit penalty with recency decay -----
        # APPLY ONLY IF WE MOVED TO A NEW CELL (prevents heavy penalty on first turn-in-place)
        if moved_to_new_cell and replay_buffer:
            for age, state in enumerate(islice(reversed(replay_buffer), 0, window)):
                pose = state.get("imagined_pose") or state.get("real_pose")
                if pose is None:
                    continue
                try:
                    px, py = int(pose[0]), int(pose[1])
                except Exception:
                    continue
                if (px, py) == (x, y):
                    pen = max(pen, float(revisit_large * math.exp(-age / tau)))
                    break  # most recent hit dominates

        return float(pen)
    def get_memory_map_data(self, dbg=True):
        if self.cog is None:
            return {'exps_GP': [], 'exps_decay': [], 'ghost_exps_GP': [],
                    'ghost_exps_link': [], 'exps_links': [], 'current_exp_id': -1}

        mg   = self.cog.mg
        emap = mg.experience_map

        memory_map_data = {
            'exps_GP': [], 'exps_decay': [],
            'ghost_exps_GP': [], 'ghost_exps_link': [],
            'exps_links': []
        }

        memory_map_data['current_exp_id'] = mg.get_current_exp_id()
        memory_map_data['current_GP']     = mg.get_global_position()
        if memory_map_data['current_exp_id'] < 0:
            if dbg: print("[DBG] No current experience → empty map")
            return memory_map_data
        memory_map_data['current_exp_GP'] = mg.get_exp_global_position()

        # Experiences as nodes (colored by view-cell decay)
        for vc in mg.view_cells.cells:
            for exp in vc.exps:
                memory_map_data['exps_GP'].append([exp.x_m, exp.y_m])
                memory_map_data['exps_decay'].append(vc.decay)

        # Ghost experiences
        for ghost in emap.ghost_exps:
            memory_map_data['ghost_exps_GP'].append([ghost.x_m, ghost.y_m])
            for link in ghost.links:
                memory_map_data['ghost_exps_link'] += [
                    [ghost.x_m, ghost.y_m], [link.target.x_m, link.target.y_m]
                ]

        # Real links (dedup happens after collection)
        for exp in emap.exps:
            for link in exp.links:
                if not getattr(link.target, 'ghost_exp', False):
                    memory_map_data['exps_links'] += [
                        [link.target.x_m, link.target.y_m],
                        [exp.x_m,          exp.y_m]
                    ]

        # Deduplicate consecutive link pairs
        clean, seen = [], set()
        pts = memory_map_data['exps_links']
        for i in range(0, len(pts), 2):
            p0, p1 = tuple(pts[i]), tuple(pts[i+1])
            if (p0, p1) not in seen:
                seen.add((p0, p1))
                clean.extend([list(p0), list(p1)])
        memory_map_data['exps_links'] = clean
        return memory_map_data
    def plot_cog_map(self, ax=None, dbg=False):
        """
        Build memory_map_data from the live CognitiveGraph
        and draw it using navigation_model.visualisation_tools.plot_memory_map.
        """
        import matplotlib.pyplot as plt
        from navigation_model.visualisation_tools import plot_memory_map

        data = self.get_memory_map_data(dbg=dbg)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        plot_memory_map(ax, data, dbg=dbg)  # defensive & annotated
        return ax  # return for caller to show() or embed in their figure
    def save_cog_map_snapshot(self, t_step: int, out_dir: str = "dbg/cogmap",
                          pad: float = 1.25, min_span: float | None = None,
                          dpi: int = 160, annotate_ids: bool = True):
        """
        Save a PNG of the current cognitive map (nodes + links) at:
        {out_dir}/cogmap_t{t_step:03d}.png

        - Uses the project's plotter for consistency.
        - Centers the view and enforces a minimum span so early maps aren't a tiny dot.
        - Draws a dashed rectangle around the current view.
        """
        import os
        from pathlib import Path
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # --- sanity ---
        if getattr(self, "cog", None) is None or getattr(self.cog, "mg", None) is None:
            print("[save_cog_map_snapshot] cognitive graph not initialized (planner.cog.mg is None).")
            return None

        # Build the structure your plotter expects (your own helper)
        data = self.get_memory_map_data(dbg=False)
        if data.get("current_exp_id", -1) < 0 and not data.get("exps_GP"):
            print("[save_cog_map_snapshot] nothing to plot yet; skipping.")
            return None

        # --- gather xy points robustly (drop theta if present) ---
        def _xy(p):
            try:
                # p could be (x,y), (x,y,th), numpy array, etc.
                if p is None:
                    return None
                if isinstance(p, (list, tuple, np.ndarray)):
                    if len(p) >= 2:
                        return float(p[0]), float(p[1])
            except Exception:
                pass
            return None

        xy = []

        for p in data.get("exps_GP", []):
            q = _xy(p);  xy.append(q) if q else None
        for p in data.get("ghost_exps_GP", []):
            q = _xy(p);  xy.append(q) if q else None

        # include link endpoints too (sometimes you have links before vc-decay-filled nodes)
        L = data.get("exps_links", [])
        for i in range(0, len(L), 2):
            q0 = _xy(L[i]);   q1 = _xy(L[i+1]) if i+1 < len(L) else None
            if q0: xy.append(q0)
            if q1: xy.append(q1)

        # include current markers
        q = _xy(data.get("current_exp_GP"));  xy.append(q) if q else None
        q = _xy(data.get("current_GP"));      xy.append(q) if q else None

        if not xy:
            print("[save_cog_map_snapshot] no xy points assembled; skipping.")
            return None

        P = np.asarray(xy, dtype=float)  # (N,2)

        # --- compute nice view bounds (center + margin) ---
        xmin, ymin = P.min(axis=0)
        xmax, ymax = P.max(axis=0)
        cx, cy     = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5

        # set a sensible minimum span (use map size if available, else 6.0)
        emap      = self.cog.mg.experience_map
        default_min = max(6.0, float(getattr(emap, "DIM_XY", 10)) * 0.5)
        span_min  = default_min if min_span is None else float(min_span)

        span = max(xmax - xmin, ymax - ymin, span_min)
        half = 0.5 * span * float(pad)

        # --- draw with the project's plotter for consistent style ---
        try:
            from navigation_model.visualisation_tools import plot_memory_map
            fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=dpi)
            plot_memory_map(ax, data, dbg=False)
        except Exception as e:
            # Fallback (very simple but still useful)
            fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=dpi)
            ax.set_title("Experience Map")
            ax.set_aspect("equal", "box")
            ax.grid(True)
            xs = [p[0] for p in P]; ys = [p[1] for p in P]
            ax.scatter(xs, ys, s=50, edgecolor="k", linewidth=0.4)

            # draw links if present
            for i in range(0, len(L), 2):
                q0 = _xy(L[i]); q1 = _xy(L[i+1]) if i+1 < len(L) else None
                if q0 and q1:
                    ax.plot([q0[0], q1[0]], [q0[1], q1[1]], linewidth=1.0, alpha=0.8)

        # enforce our view window and frame rectangle
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        rect = Rectangle((cx - half, cy - half), 2*half, 2*half,
                        fill=False, linewidth=1.0, linestyle="--", color="0.6")
        ax.add_patch(rect)

        # emphasize current experience
        cur_gp = _xy(data.get("current_exp_GP"))
        if cur_gp is not None:
            x0, y0 = cur_gp
            ax.plot([x0], [y0], marker="x", color="r", markersize=8, mew=2)
            if annotate_ids:
                ax.text(x0, y0, str(data.get("current_exp_id", "")),
                        color="red", fontsize=9, weight="bold", ha="left", va="bottom")

            # facing arrow if we can
            try:
                cur = getattr(emap, "current_exp", None)
                if cur is not None:
                    import math
                    dx, dy = math.cos(cur.facing_rad), math.sin(cur.facing_rad)
                    s = 0.8
                    ax.arrow(cur.x_m, cur.y_m, s*dx, s*dy,
                            width=0.02, head_width=0.25, head_length=0.25,
                            color="tab:red", length_includes_head=True, alpha=0.7)
            except Exception:
                pass

        # annotate *all* node ids if requested (helpful early on)
        if annotate_ids and getattr(emap, "exps", None):
            for e in emap.exps:
                ax.text(e.x_m, e.y_m, f"{e.id}", fontsize=7, color="0.25", ha="center", va="center")

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f"cogmap_t{t_step:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[save_cog_map_snapshot] saved {out_path}")
        return out_path
    def _resolve_memory_graph(self, root=None):
        """
        Try to find the same graph object your viz uses.
        Looks in the common places; returns the first non-None.
        """
        candidates = [
            getattr(self, "_last_memory_graph", None),
            getattr(self, "memory_graph", None),
            getattr(root, "memory_graph", None),

            # common containers
            getattr(getattr(self, "navigation_model", None), "memory_graph", None),
            getattr(getattr(self, "navigation_model", None), "graph", None),
            getattr(self, "experience_map", None),
            getattr(self, "emap", None),
            getattr(self, "exp_map", None),
            getattr(self, "cognitive_graph", None),
            getattr(self, "cog_graph", None),
        ]
        for g in candidates:
            if g is not None:
                return g
        return None


    def _collect_graph_nodes_and_weights(self, memory_graph):
        """
        Returns (node_xy_list, weight_list). If weights unavailable, returns ones.
        Mirrors the patterns viz code typically uses, and falls back to lots of
        common graph shapes (NetworkX, dicts, lists of objects, experience maps).
        Applies optional coord transform hook if provided.
        """
        out_xy, out_w = [], []

        # Optional coord transform hook to match your viz frame
        coord_tf = getattr(self, "graph_coord_transform", None)
        def _tf(xy):
            if callable(coord_tf):
                try:
                    return tuple(map(int, coord_tf(tuple(xy))))
                except Exception:
                    pass
            return (int(xy[0]), int(xy[1]))

        # Optional explicit provider (strongest signal)
        provider = getattr(self, "graph_nodes_provider", None)
        if callable(provider):
            try:
                items = list(provider())
                for it in items:
                    if isinstance(it, (tuple, list)):
                        if len(it) >= 3: x, y, w = it[0], it[1], float(it[2])
                        elif len(it) >= 2: x, y, w = it[0], it[1], 1.0
                        else: continue
                        out_xy.append(_tf((x, y))); out_w.append(w)
                if out_xy: return out_xy, out_w
            except Exception:
                pass

        G = memory_graph
        if G is None:
            return [], []

        # Unwrap one layer (.G or .graph) — matches how many viz funcs stash the nx graph
        for attr in ("G", "graph"):
            if hasattr(G, attr):
                try:
                    inner = getattr(G, attr)
                    if inner is not None:
                        G = inner
                        break
                except Exception:
                    pass

        # --- NetworkX-like ---
        try:
            # (a) nodes(data=True)
            try:
                got = False
                for _, data in G.nodes(data=True):
                    if not data: 
                        continue
                    xy = None
                    if "x" in data and "y" in data:               xy = (data["x"], data["y"])
                    elif "pose" in data and len(data["pose"])>=2: xy = (data["pose"][0], data["pose"][1])
                    elif "pos"  in data and len(data["pos"]) >=2: xy = (data["pos"][0],  data["pos"][1])
                    elif "xy"   in data and len(data["xy"])  >=2: xy = (data["xy"][0],   data["xy"][1])
                    elif "coord" in data and len(data["coord"])>=2: xy=(data["coord"][0], data["coord"][1])
                    elif "row" in data and "col" in data:         xy = (data["col"], data["row"])
                    if xy is None: 
                        continue
                    w = float(data.get("weight", 1.0))
                    out_xy.append(_tf(xy)); out_w.append(w); got = True
                if got: 
                    return out_xy, out_w
            except TypeError:
                pass

            # (b) pos maps commonly used in viz code
            try:
                # tolerant access without importing networkx
                pos_map = None
                for key in ("pos", "positions", "node_pos", "node_positions"):
                    pos_map = getattr(G, key, None)
                    if pos_map: break
                # networkx-style attribute dicts: G.nodes[n]['pos']
                if not pos_map:
                    # try reading attributes from nodes directly
                    # (if nodes() works but data=True path didn't give coords)
                    for nid in list(G.nodes()):
                        try:
                            data = G.nodes[nid]
                            if isinstance(data, dict) and "pos" in data and len(data["pos"])>=2:
                                if not pos_map: pos_map = {}
                                pos_map[nid] = data["pos"]
                        except Exception:
                            continue
                if pos_map:
                    for nid, p in pos_map.items():
                        if isinstance(p, (tuple, list)) and len(p) >= 2:
                            out_xy.append(_tf((p[0], p[1]))); out_w.append(1.0)
                    if out_xy: 
                        return out_xy, out_w
            except Exception:
                pass
        except Exception:
            pass

        # --- dict-like .nodes: dict[id] -> object/dict ---
        nodes_obj = getattr(G, "nodes", None)
        if isinstance(nodes_obj, dict):
            got = False
            for _, obj in nodes_obj.items():
                xy = None; w = 1.0
                if hasattr(obj, "x") and hasattr(obj, "y"):
                    xy = (getattr(obj, "x"), getattr(obj, "y"))
                elif hasattr(obj, "pose") and isinstance(getattr(obj, "pose"), (tuple, list)) and len(getattr(obj, "pose"))>=2:
                    p = getattr(obj, "pose"); xy = (p[0], p[1])
                elif isinstance(obj, dict):
                    if "x" in obj and "y" in obj:                   xy = (obj["x"], obj["y"])
                    elif "pose" in obj and len(obj["pose"]) >= 2:   xy = (obj["pose"][0], obj["pose"][1])
                    elif "pos" in obj and len(obj["pos"]) >= 2:     xy = (obj["pos"][0],  obj["pos"][1])
                    elif "xy" in obj and len(obj["xy"]) >= 2:       xy = (obj["xy"][0],   obj["xy"][1])
                    elif "row" in obj and "col" in obj:             xy = (obj["col"], obj["row"])
                if xy is not None:
                    out_xy.append(_tf(xy)); out_w.append(w); got = True
            if got: 
                return out_xy, out_w

        # --- arrays commonly used in viz overlays / experience maps ---
        for attr in ("experiences", "nodes", "anchors", "vertices", "points"):
            arr = getattr(G, attr, None)
            if isinstance(arr, (list, tuple)):
                got = False
                for obj in arr:
                    xy = None; w = 1.0
                    if hasattr(obj, "x") and hasattr(obj, "y"):
                        xy = (getattr(obj, "x"), getattr(obj, "y"))
                    elif hasattr(obj, "pose") and isinstance(getattr(obj, "pose"), (tuple, list)) and len(getattr(obj, "pose"))>=2:
                        p = getattr(obj, "pose"); xy = (p[0], p[1])
                    elif isinstance(obj, dict):
                        if "x" in obj and "y" in obj:                 xy = (obj["x"], obj["y"])
                        elif "pose" in obj and len(obj["pose"]) >= 2: xy = (obj["pose"][0], obj["pose"][1])
                        elif "pos" in obj and len(obj["pos"]) >= 2:   xy = (obj["pos"][0],  obj["pos"][1])
                        elif "xy" in obj and len(obj["xy"]) >= 2:     xy = (obj["xy"][0],   obj["xy"][1])
                        elif "row" in obj and "col" in obj:           xy = (obj["col"], obj["row"])
                    if xy is not None:
                        out_xy.append(_tf(xy)); out_w.append(w); got = True
                if got: 
                    return out_xy, out_w

        # --- raw iterable of tuples/lists (sometimes viz pipelines build these) ---
        if isinstance(G, (list, tuple)):
            try:
                for it in G:
                    if isinstance(it, (tuple, list)):
                        if len(it) >= 3: x, y, w = it[0], it[1], float(it[2])
                        elif len(it) >= 2: x, y, w = it[0], it[1], 1.0
                        else: continue
                        out_xy.append(_tf((x, y))); out_w.append(w)
                if out_xy:
                    return out_xy, out_w
            except Exception:
                pass

        return [], []

    # --- 2) Save an overlay of env + cognitive graph, SPAWN-anchored (translation only) ---
    def save_cog_map_vs_env(
        self,
        env,
        t_step: int,
        out_dir: str = "dbg/cogmap",
        tile_size: int = 32,
        annotate_ids: bool = True,
        dpi: int = 160,
    ):
        """
        Snapshot the environment image and the cognitive graph *separately* for
        later, flexible overlay in an external script. No in-loop alignment/overlay.

        Outputs (inside {out_dir}/t{t_step:04d}/):
        - env_tXXXX.png           : full environment render
        - cog_tXXXX.png           : transparent PNG of just the cognitive graph
        - snapshot_tXXXX.json     : metadata linking both + cog geometry/bbox

        The JSON contains:
        {
            "t_step": int,
            "timestamp": ISO8601,
            "env": {
            "img_path": str,
            "pixel_size": [H, W],
            "grid_size": [Wc, Hc],   # grid cells
            "tile_size": int
            },
            "cog": {
            "png_path": str,         # transparent graph image
            "nodes": [{"id": int|None, "x": float, "y": float,
+            "decay": float|None, "real_pose": [x,y,dir]|None,
+            "place_kind": "ROOM"|"CORRIDOR"|"UNKNOWN"|None,
+            "room_color": str|None,           # e.g., "purple"
+            "grid_xy": [gx,gy]|None}, ...],
            "links_xy": [[x1,y1],[x2,y2], ...],  # pairs, same order as drawn
            "current_exp_id": int|None,
            "bbox": [minx, maxx, miny, maxy]     # native cog coordinate bbox
            },
            "spawn": {"x": int, "y": int, "dir": int}|null
        }

        Later you can rotate/mirror/scale/translate the cog layer over the env image.
        """
        import os, json, math, datetime
        from pathlib import Path
        import numpy as np

        # --- Render env image ---
        try:
            env_img = env.render(mode="rgb_array")
        except TypeError:
            env_img = env.render("rgb_array", tile_size=tile_size)

        H, W = env_img.shape[:2]
        # Best-effort grid size from env if present; fallback to pixel/tile
        Wc = int(getattr(env, "width",  max(1, W // max(1, tile_size))))
        Hc = int(getattr(env, "height", max(1, H // max(1, tile_size))))

        # --- Access cognitive graph safely ---
        if not getattr(self, "cog", None) or not getattr(self.cog, "mg", None):
            print("[save_cog_map_vs_env] WARN: cognitive graph not initialized; saving env only.")
            cog_available = False
            mg = None
            emap = None
        else:
            cog_available = True
            mg = self.cog.mg
            emap = getattr(mg, "experience_map", None)
            if emap is None:
                cog_available = False
                print("[save_cog_map_vs_env] WARN: mg.experience_map missing; saving env only.")

        # --- Snapshot folder ---
        snap_dir = Path(out_dir) / f"t{t_step:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # --- Save ENV image ---
        # Use matplotlib's Agg to avoid display backends
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        env_png = snap_dir / f"env_t{t_step:04d}.png"
        fig_env, ax_env = plt.subplots(1, 1, figsize=(W/80, H/80), dpi=80)  # keep pixels 1:1-ish
        ax_env.imshow(env_img, origin="upper")
        ax_env.axis("off")
        fig_env.savefig(env_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig_env)

        # --- Build cognitive graph geometry ---
        nodes = []
        links_xy = []
        current_exp_id = None
        spawn_pose = None
        bbox = None
        cog_png = None

        if cog_available:
            # Prefer your own helper if present
            try:
                data = self.get_memory_map_data(dbg=False)
            except Exception:
                data = {}

            # Nodes: try emap.exps (with ids & real_pose) as the authoritative source
            exps = list(getattr(emap, "exps", []) or [])
            have_exps_objects = len(exps) > 0

            # Fallback to data['exps_GP'] if no objects
            if have_exps_objects:
                # Build a decay lookup if available (align by id or by order)
                decays = data.get("exps_decay", None)
                # If decay length matches number of exps, align by index; else None
                decay_by_index = decays if isinstance(decays, (list, tuple)) and len(decays) == len(exps) else None

                for idx, e in enumerate(exps):
                    rp  = getattr(e, "real_pose", None)
                    gxy = getattr(e, "grid_xy", None)
                    if isinstance(gxy, (list, tuple)) and len(gxy) == 2:
                        try:
                            gxy_out = [int(gxy[0]), int(gxy[1])]
                        except Exception:
                            gxy_out = [float(gxy[0]), float(gxy[1])]
                    else:
                        gxy_out = None

                    nodes.append({
                        "id": int(getattr(e, "id", idx)) if getattr(e, "id", None) is not None else None,
                        "x": float(getattr(e, "x_m", 0.0)),
                        "y": float(getattr(e, "y_m", 0.0)),
                        "decay": float(decay_by_index[idx]) if decay_by_index is not None else None,
                        "real_pose": [int(rp[0]), int(rp[1]), int(rp[2])] if (isinstance(rp, (list, tuple)) and len(rp) >= 3) else None,
                        # --- NEW semantic fields ---
                        "place_kind": getattr(e, "place_kind", None),
                        "room_color": getattr(e, "room_color", None),
                        "grid_xy": gxy_out,
                    })
            else:
                # Use the raw GP list if that's what you maintain
                # Use the raw GP list if that's what you maintain
                gps = data.get("exps_GP", []) or []
                decays = data.get("exps_decay", [])
                for i, (x, y) in enumerate(gps):
                    dec = float(decays[i]) if i < len(decays) else None
                    nodes.append({
                        "id": None,
                        "x": float(x),
                        "y": float(y),
                        "decay": dec,
                        "real_pose": None,
                        # --- semantic fields unavailable in this fallback ---
                        "place_kind": None,
                        "room_color": None,
                        "grid_xy": None,
                    })
            # Links: use coords as saved by your helper
            raw_links = data.get("exps_links", []) or []
            # Ensure they are pairs of points [[x1,y1],[x2,y2], [x3,y3],[x4,y4], ...]
            if len(raw_links) >= 2:
                for i in range(0, len(raw_links) - 1, 2):
                    a = raw_links[i]
                    b = raw_links[i + 1]
                    if a is None or b is None or len(a) < 2 or len(b) < 2:
                        continue
                    links_xy.append([float(a[0]), float(a[1])])
                    links_xy.append([float(b[0]), float(b[1])])

            # Current node id if available
            current_exp_id = data.get("current_exp_id", None)
            try:
                current_exp_id = int(current_exp_id) if current_exp_id is not None else None
            except Exception:
                current_exp_id = None

            # Spawn (best-effort)
            spawn_pose = getattr(emap, "spawn_pose_real", None)
            if spawn_pose is None:
                # fallback to current agent pose if env provides it
                try:
                    spawn_pose = (int(env.agent_pos[0]), int(env.agent_pos[1]), int(getattr(env, "agent_dir", 0)))
                    print("[save_cog_map_vs_env] INFO: using env.agent_pos as spawn fallback.")
                except Exception:
                    spawn_pose = None

            # Compute bbox in *native cog coords*
            if nodes:
                xs = [n["x"] for n in nodes]
                ys = [n["y"] for n in nodes]
                minx, maxx = float(min(xs)), float(max(xs))
                miny, maxy = float(min(ys)), float(max(ys))
                # Add a small margin so the PNG isn't tight against the edges
                mx = (maxx - minx) * 0.05 if maxx > minx else 0.5
                my = (maxy - miny) * 0.05 if maxy > miny else 0.5
                bbox = [minx - mx, maxx + mx, miny - my, maxy + my]
            else:
                bbox = [0.0, 1.0, 0.0, 1.0]  # dummy, still usable

            # --- Save transparent COG PNG ---
            cog_png = snap_dir / f"cog_t{t_step:04d}.png"
            fig_c, ax_c = plt.subplots(1, 1, dpi=dpi)
            # Transparent everything
            fig_c.patch.set_alpha(0.0)
            ax_c.set_facecolor((1, 1, 1, 0.0))

            # Plot links first
            if links_xy:
                for i in range(0, len(links_xy) - 1, 2):
                    x1, y1 = links_xy[i]
                    x2, y2 = links_xy[i + 1]
                    ax_c.plot([x1, x2], [y1, y2], color="k", linewidth=1.2, alpha=0.9)

            # Plot nodes
            if nodes:
                nxy = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
                ax_c.scatter(nxy[:, 0], nxy[:, 1],
                            s=50, c="tab:blue", edgecolors="k", linewidths=0.5, alpha=0.95)
                if annotate_ids:
                    for n in nodes:
                        if n["id"] is not None:
                            ax_c.text(n["x"], n["y"], f"{n['id']}", fontsize=7,
                                    ha="center", va="center", color="0.2", alpha=0.9)

            # Highlight current node if we have it
            if current_exp_id is not None and nodes:
                # try to find coords for current id
                for n in nodes:
                    if n["id"] == current_exp_id:
                        ax_c.plot([n["x"]], [n["y"]], marker="x", color="r", mew=2, ms=9, alpha=0.95)
                        break

            # Use bbox so the image aligns with native cog units
            ax_c.set_xlim(bbox[0], bbox[1])
            ax_c.set_ylim(bbox[2], bbox[3])
            ax_c.set_aspect("equal", "box")
            ax_c.axis("off")
            fig_c.savefig(cog_png, transparent=True, bbox_inches="tight", pad_inches=0)
            plt.close(fig_c)

        # --- Write JSON snapshot manifest ---
        manifest = {
            "t_step": int(t_step),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "env": {
                "img_path": str(env_png),
                "pixel_size": [int(H), int(W)],
                "grid_size": [int(Wc), int(Hc)],
                "tile_size": int(tile_size),
            },
            "cog": {
                "png_path": str(cog_png) if cog_png else None,
                "nodes": nodes,
                "links_xy": links_xy,           # drawn as pairs [A,B],[C,D],...
                "current_exp_id": current_exp_id,
                "bbox": bbox,
            },
            "spawn": {"x": int(spawn_pose[0]), "y": int(spawn_pose[1]), "dir": int(spawn_pose[2])} if spawn_pose else None,
        }

        man_path = snap_dir / f"snapshot_t{t_step:04d}.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[save_cog_map_vs_env] Snapshot saved:\n  {env_png}\n  {cog_png if cog_png else '(no cog)'}\n  {man_path}")
        return str(man_path)

    




import gym
import cv2
class DictResizeObs(gym.ObservationWrapper):
    def __init__(self, env, out_hw=(64,64)):
        super().__init__(env)
        self.out_hw = out_hw              # (H,W)

    def observation(self, obs):
        assert isinstance(obs, dict) and "image" in obs, \
               "expect dict with 'image' key"
        img = obs["image"]                               # (H,W,3) uint8
        img = cv2.resize(img, self.out_hw[::-1],
                         interpolation=cv2.INTER_AREA)
        obs["image"] = img
        return obs

if __name__ == "__main__":
    import time, random, numpy as np, torch
    import gym, gym_minigrid
    from collections import deque
    from itertools import islice
    from gym_minigrid.minigrid import Wall
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgActionObsWrapper
    from world_model_utils import DictResizeObs, WMPlanner, State

    # ---------- config ----------
    CKPT        = "runs/mg_collision/20250704-220917/ckpt/iter05000.pt"
    N_STEPS     = 1290            # run the novelty policy for this many env steps
    LOOKAHEAD   = 7             # A* novelty horizon
    K_RECENT    = 30            # how many recent embeddings to compare against
    METRIC      = "cos"         # "kl" | "cos" | "l2"
    DEVICE      = torch.device("cpu")

    # Debug/vis knobs
    DEBUG_PRINT          = True   # keep tree/top-K/action scores logs
    VIZ_TREE             = False   # save decision tree collage from novelty_astar_plan
    VIZ_EVERY            = 15      # save imagined rollout strip and replay history every k steps (set 1 to save each step)
    VIZ_OUTDIR           = "dbg"  # where to write images
    SAVE_HISTORY_LENGTHS = (30, 60, 90, 120,150,180,210,240,270,300,330,360,390,410,440,470,500,530,560,590,620,650,685,710,740,760,790,890,990,1090,1190,1290)

    # ---------- boot world model & env --------
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]  # repo root (…/hierarchical-nav/)
    MEMCFG = ROOT / "navigation_model/Services/memory_service/memory_graph_config.yml"

    planner = WMPlanner(ckpt=CKPT,device=str(DEVICE),memory_config=str(MEMCFG))
    wm = planner.wm
    print("✓ world-model loaded")

    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0", rooms_in_row=7, rooms_in_col=7, max_steps=None)
    env.seed(218)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env)
    env = DictResizeObs(env, (64, 64))
    planner.env=env
    
    # ---------- helpers ----------
    def decode_policy_onehot(onehot_tensor: torch.Tensor) -> list[str]:
        """
        Fallback decoder if novelty_astar_plan returns a (T,3) plan.
        """
        idx_to_act = {0: "forward", 1: "right", 2: "left"}
        if onehot_tensor is None or onehot_tensor.numel() == 0:
            return []
        acts = []
        for row in onehot_tensor.cpu():
            j = int(row.argmax().item())
            acts.append(idx_to_act[j])
        return acts

    def dreamer_decode_from_belief(planner, wm, belief_zd, *, to_01: bool = True):
        import torch
        from torch.distributions import Distribution

        if belief_zd is None:
            return None

        z, h = belief_zd
        with torch.no_grad():
            out = wm.decoder(z, h)
            x = out.mean if isinstance(out, Distribution) else out
            x = torch.as_tensor(x).detach().float()
            if x.ndim == 4 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2:
                x = x.unsqueeze(0).repeat(3, 1, 1)
            elif x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1).contiguous()
            elif x.ndim == 3 and x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if to_01:
                x = (x + 0.5).clamp(0.0, 1.0)
            return x.cpu()

    def append_to_replay_buffer(buf: deque, obs: dict, action_int: int, belief_zd):
        """
        Store Dreamer belief, embeddings, decoded prediction, and enhanced predictions.
        """
        z_embed = planner.dreamer_embed_fn(belief_zd)  # (Z,) flattened image embedding in [0,1]
        decoded = dreamer_decode_from_belief(planner, wm, belief_zd)  # (3,H,W) or None
        enhanced_preds = planner.build_enhanced_perception(
            wm, belief_zd,
            combos=[('left',), ('right',), ('left','left'), ('right','right')]
        )
        entry = {
            "node_id": None,
            "real_pose": obs.get("pose"),
            "imagined_pose": None,
            "real_image": obs.get("image"),
            "imagined_image": None,
            "action": int(action_int),
            "dreamer_z": z_embed,
            "belief_zd": belief_zd,
            "decoded_image": decoded,
            "enhanced_preds": enhanced_preds,
        }
        buf.append(entry)

    def visualize_policy_probe(planner, wm, belief_zd, obs, action_names, t_step,
                               out_dir="dbg", include_start=True):
        import torch
        from pathlib import Path
        from torchvision.utils import save_image

        if not action_names:
            return None

        real_frame = torch.as_tensor(obs["image"]).permute(2, 0, 1).float() / 255.0 - 0.5
        frames_pred = planner.render_plan(wm, belief_zd, action_names, include_last=True)
        frames = [real_frame.detach().cpu()] + [f.detach().cpu() for f in frames_pred]
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        short = "".join(a[0].upper() for a in action_names)
        out_path = f"{out_dir}/probe_t{t_step:02d}_{short}.png"
        save_image(torch.stack(frames), out_path, nrow=len(frames), normalize=True, scale_each=False)
        print(f"saved {out_path}")
        return out_path

    def save_replay_image_history(replay_buffer, planner, wm, t_step, Ns=(5,10,15,20), out_dir="dbg/history"):
        import torch
        from pathlib import Path
        from torchvision.utils import save_image

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        def to_chw_real(img_hwc):
            t = torch.as_tensor(img_hwc).permute(2,0,1).float() / 255.0 - 0.5
            return t

        for N in Ns:
            tail = list(islice(reversed(replay_buffer), 0, N))
            if not tail:
                continue
            tail = list(reversed(tail))

            reals, preds = [], []
            for st in tail:
                ri = st.get("real_image", None)
                bi = st.get("belief_zd", None)
                if ri is None:
                    continue
                reals.append(to_chw_real(ri))
                di = st.get("decoded_image", None)
                if di is None and bi is not None:
                    di = dreamer_decode_from_belief(planner, wm, bi)
                if di is None:
                    di = reals[-1]
                preds.append(di)

            if not reals:
                continue

            frames = reals + preds
            out_path = f"{out_dir}/history_t{t_step:02d}_N{len(reals)}.png"
            save_image(torch.stack(frames), out_path, nrow=len(reals), normalize=True, scale_each=False)
            print(f"saved {out_path}")

    
    # ---------- init replay & belief ----------
    replay_buffer = deque(maxlen=30)
    obs = env.reset()
    # --- at episode start
    episode_id=1
    collage = DualPathCollage(
        out_dir="dbg/collages",
        tag=f"ep{episode_id:03d}",
        tile=64,          # match your Dreamer decode size if 64x64
        cols=20,          # tweak to taste
        live_write=True,  # set False if you only want the final PNGs
        live_every=1,     # write after every step when live
        annotate_idx=True,
        planner=planner,  # so it can call dreamer_decode_from_belief
        wm=wm
    )


    # bootstrap belief from first observation
    frame = obs["image"].transpose(2,0,1) / 255.0 - 0.5
    belief = planner.wm_update_belief(wm, prev_z_d=None, frame_rgb=frame, prev_action_onehot=None)

    # take one forward step to match your current flow, then update belief and seed buffer
    obs, _, done, _ = env.step(env.actions.forward)
    prev_act_1h = planner.onehot("forward", wm)
    belief = planner.update_belief_from_obs(obs, belief, prev_act_1h)
    append_to_replay_buffer(replay_buffer, obs, env.actions.forward, belief)
    pose_xyz = tuple(obs["pose"])
    print(obs["image"].shape)
    collage.add_step(real_img=obs["image"], belief_zd=belief)

    planner.update_cog(obs["image"], prev_act_1h,pose_xyz, place_post=None)
    print(planner.get_cog_nodes())

    # maps between env ints and names
    act_to_name = {
        env.actions.left: "left",
        env.actions.right: "right",
        env.actions.forward: "forward",
    }
    name_to_act = {v: k for k, v in act_to_name.items()}

    def choose_action_from_novastar(result):
        """
        Accepts either:
          • (best_action: str, scores: dict, top_paths: list)     ← current utils
          • torch.Tensor one-hot plan (T,3), or list[str] plan    ← older utils
        Returns: (action_name: str, voted_scores: dict[str,float] | None)
        """
        import torch
        # Newer signature: tuple where first item is a string action name
        if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
            best_action = result[0]
            scores = result[1] if len(result) >= 2 and isinstance(result[1], dict) else None
            return best_action, scores

        # Older signature: full plan
        if isinstance(result, torch.Tensor):
            seq = decode_policy_onehot(result)
        elif isinstance(result, (list, tuple)) and all(isinstance(a, str) for a in result):
            seq = list(result)
        else:
            seq = []

        if seq:
            return seq[0], None
        return None, None

    print("\n=== Novelty-driven rollout (every step) ===")
    for t in range(1, N_STEPS + 1):
        start = State(*obs["pose"])

        # --- run exploration algorithm to pick ONE next action ---
        result = planner.novelty_astar_plan(
            wm=wm,
            belief_zd=belief,
            start_state=start,
            lookahead=LOOKAHEAD,
            replay_buffer=replay_buffer,
            K_recent=K_RECENT,
            metric=METRIC,
            verbose=DEBUG_PRINT,
            viz_tree=VIZ_TREE,
            viz_outdir=VIZ_OUTDIR,
            viz_tag=f"t{t:02d}",
            # weights / uncertainty-rollout knobs
            w_p2e=1.0,             # turn on K-rollout cost
            p2e_K=5,
            p2e_mode="temporal",   # your requested variant
            p2e_metric="l2",
            p2e_roll_from="parent" # start from node belief and apply 'act' K times
        )

        action_name, scores = choose_action_from_novastar(result)

        # Fallback if planner returns nothing:
        if action_name is None:
            # try the _last_action_scores if available
            scores_attr = getattr(planner, "_last_action_scores", None)
            if isinstance(scores_attr, dict) and scores_attr:
                action_name = max(scores_attr.items(), key=lambda kv: kv[1])[0]
            else:
                # last-resort, simple "don't collide" heuristic
                front_pos = env.front_pos
                front_cell = env.grid.get(*front_pos)
                action_name = "left" if isinstance(front_cell, Wall) else "forward"


        env_act = name_to_act[action_name]

        # --- take the step ---
        next_obs, _, done, _ = env.step(env_act)

        # --- update belief with observed frame and prev action ---
        prev_act_1h = planner.onehot(action_name, wm)
        belief = planner.update_belief_from_obs(next_obs, belief, prev_act_1h)
        pose_xyz = tuple(next_obs["pose"])
        collage.add_step(real_img=obs["image"], belief_zd=belief)
        planner.update_cog(next_obs["image"], prev_act_1h,pose_xyz, place_post=None)
        if planner.emap.current_exp is not None:
            e = planner.emap.current_exp
            print(f"[PLACE] Exp{e.id} at {e.grid_xy}: {e.place_kind}"
                + (f" ({e.room_color})" if e.room_color else ""))
        # --- push transition to replay buffer (includes decoded prediction) ---
        append_to_replay_buffer(replay_buffer, next_obs, env_act, belief)

        # --- debug prints / tree / top-K paths ---
        if DEBUG_PRINT:
            print(f"[t={t:02d}] chose action → {action_name}")
            # if internal records exist, echo them (kept from your probe block)
            tree = getattr(planner, "_last_decision_tree", None)
            if hasattr(planner, "_last_top_paths"):
                print("\n=== TOP-K PATHS USED FOR VOTING ===")
                for i, rec in enumerate(planner._last_top_paths, 1):
                    a0 = rec["seq"][0] if rec["seq"] else "∅"
                    print(f"#{i:02d} G={rec['gain']:.3f} first={a0:<7} depth={rec['depth']} "
                          f"seq={rec['seq']} pruned={rec.get('pruned_reason')}")
            if hasattr(planner, "_last_action_scores"):
                print("Action vote scores:", getattr(planner, "_last_action_scores"))

        # --- visualizations (policy rollout + history strips) ---
        if (t % max(1, VIZ_EVERY)) == 0:
            # visualize 1-step policy just for local context; if you want longer, call
            # planner.rank_paths(...) to grab the best sequence and pass it in.
            visualize_policy_probe(planner, wm, belief, next_obs, [action_name], t_step=t,
                                   out_dir=VIZ_OUTDIR)
            save_replay_image_history(replay_buffer, planner, wm, t_step=t,
                                      Ns=SAVE_HISTORY_LENGTHS, out_dir=f"{VIZ_OUTDIR}/history")
            planner.save_cog_map_snapshot(t_step=t, out_dir=f"{VIZ_OUTDIR}/cogmap")
            planner.save_cog_map_vs_env(env, t_step=t, out_dir="dbg/cogmap", tile_size=32, dpi=160)
            
            # advance obs pointer
        obs = next_obs

        # --- handle episode end robustly ---
        if done:
            obs = env.reset()
            frame = obs["image"].transpose(2,0,1) / 255.0 - 0.5
            belief = planner.wm_update_belief(wm, prev_z_d=None, frame_rgb=frame, prev_action_onehot=None)
            obs, _, done, _ = env.step(env.actions.forward)
            prev_act_1h = planner.onehot("forward", wm)
            belief = planner.update_belief_from_obs(obs, belief, prev_act_1h)
            replay_buffer.clear()
            collage.finalize()
            append_to_replay_buffer(replay_buffer, obs, env.actions.forward, belief)

    print("\nDone. (Executed novelty policy at every step.)")

""" def first_action_from_topk_paths(
        self,
        root,
        *,
        k: int = 10,
        include_pruned: bool = True,
        weight: str = "linear",     # "linear" | "harmonic" | "exp"
        alpha: float = 0.85,        # only used when weight="exp"
        require_min_depth: int = 1, # ignore empty sequences (no first action)
        verbose: bool = False,
    ):
        """ 
"""         Rank all root→terminal paths by a NEW score that ignores the builder's 'gain':
            score(path) = new_gain_from_steps
                        + B * dist_from_start
                        + C * dist_from_graph_nodes   (stub; 0.0 if not wired)

        new_gain_from_steps uses only per-step metrics stored in TreeNode:
            raw_nov      ≔ step_novelty + step_penalty
            p2e_bonus    ≔ inverted, sibling-normalized step_p2e in [0,1]
            step_value   ≔ (raw_nov - step_penalty) + Wp2e * p2e_bonus
            new_gain     ≔ Σ_t (γ_rank)^(t-1) * step_value_t

        Everything else (top-k, rank-weighted first-action voting) is unchanged.
        Returns:
            best_action : str   in {"left","right","forward"}
            action_scores : dict[str, float]
            top_paths : list[dict] enriched with 'score' and 'score_terms' """ """

        # -------------------------------
        # Tunables (readable from self.* if present; fall back to defaults)
        # -------------------------------
        Wp2e   = getattr(self, "rank_Wp2e", 1.0)         # weight for sibling-normalized p2e bonus in step_value
        gammaR = getattr(self, "rank_gamma", 1.0)        # per-step discount in new_gain_from_steps
        Bdist  = getattr(self, "rank_B", 0.10)           # B * distance from start
        Cgraph = getattr(self, "rank_C", 0.00)           # C * distance to graph nodes (stub—computed as 0.0 unless available)
        dist_metric = getattr(self, "rank_dist_metric", "manhattan")  # or "euclidean"

        # -------------------------------
        # Utilities: traverse the tree along a seq, compute sibling p2e stats, and distances
        # -------------------------------
        def _path_nodes_by_seq(root_node, seq):
            
            nodes = []
            cur = root_node
            prefix = []
            for a in seq:
                prefix.append(a)
                nxt = None
                # prefer exact-seq match for disambiguation
                for ch in cur.children:
                    if getattr(ch, "seq", None) == prefix:
                        nxt = ch
                        break
                if nxt is None:
                    # fallback: first child matching the action
                    for ch in cur.children:
                        if getattr(ch, "step_act", None) == a:
                            nxt = ch
                            break
                if nxt is None:
                    return []  # broken path
                nodes.append(nxt)
                cur = nxt
            return nodes

        def _sibling_p2e_norm_inverted(child_node, parent_node):
            
            try:
                sibs = [c for c in getattr(parent_node, "children", []) if hasattr(c, "step_p2e")]
                vals = [float(c.step_p2e) for c in sibs if (c.step_p2e is not None) and (float(c.step_p2e) >= 0.0)]
                v = float(child_node.step_p2e)
                if not vals or len(vals) == 1:
                    return 0.5
                vmin, vmax = min(vals), max(vals)
                if vmax <= vmin:
                    return 0.5
                # invert: best (smallest) p2e → 1.0 ; worst (largest) → 0.0
                return (vmax - v) / (vmax - vmin + 1e-8)
            except Exception:
                return 0.5

        def _dist(p0, p1):
            (x0, y0, _d0) = p0
            (x1, y1, _d1) = p1
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            if dist_metric == "euclidean":
                return (dx * dx + dy * dy) ** 0.5
            return dx + dy  # manhattan default

        def _graph_distance_stub(node_pose):
            """""" Stub for future graph metrics. If you later wire a function like:
                self.distance_to_graph_nodes(pose) -> float
            we'll call it here. For now returns 0.0. """"""
            fn = getattr(self, "distance_to_graph_nodes", None)
            if callable(fn):
                try:
                    return float(fn(node_pose))
                except Exception:
                    return 0.0
            return 0.0

        # -------------------------------
        # 1) Get all paths (existing builder output)
        # -------------------------------
        ranked_all = self.rank_paths(root, include_pruned=include_pruned)  # DON'T trust 'gain' for ranking anymore

        # -------------------------------
        # 2) Compute NEW score per path (ignore 'gain')
        # -------------------------------
        start_pose = getattr(root, "pose", None) or (None, None, None)
        enriched = []
        for rec in ranked_all:
            seq   = rec.get("seq", [])
            depth = int(rec.get("depth", len(seq) or 0))
            if depth < require_min_depth:
                continue

            # follow the tree to get the actual nodes along this seq (need parent-child to do sibling norm)
            step_nodes = _path_nodes_by_seq(root, seq)
            if not step_nodes:
                # If we cannot reconstruct nodes, skip this path conservatively.
                continue

            # accumulate step contributions
            new_gain = 0.0
            parent = root
            for t, node in enumerate(step_nodes, start=1):
                # reconstruct raw_nov from stored parts
                step_pen = float(getattr(node, "step_penalty", 0.0))
                step_nov_minus_pen = float(getattr(node, "step_novelty", 0.0))
                raw_nov = step_nov_minus_pen + step_pen  # since step_novelty = raw_nov - pen

                # sibling-normalized, inverted p2e ∈ [0,1]
                p2e_bonus = _sibling_p2e_norm_inverted(node, parent)

                step_value = (raw_nov - step_pen) + (Wp2e * p2e_bonus)
                new_gain += (gammaR ** (t - 1)) * step_value
                parent = node

            # distance augments
            end_pose = rec.get("end_pose", getattr(step_nodes[-1], "pose", None))
            dist_start = _dist(start_pose, end_pose) if (start_pose[0] is not None and end_pose is not None) else 0.0
            dist_graph = _graph_distance_stub(end_pose)

            score = new_gain + Bdist * dist_start + Cgraph * dist_graph

            # stash for ranking + debugging
            rec2 = dict(rec)  # shallow copy original record
            rec2["score"] = float(score)
            rec2["score_terms"] = {
                "new_gain": float(new_gain),
                "dist_from_start": float(dist_start),
                "dist_from_graph_nodes": float(dist_graph),
                "B": float(Bdist),
                "C": float(Cgraph),
                "Wp2e": float(Wp2e),
                "gamma_rank": float(gammaR),
            }
            enriched.append(rec2)

        # -------------------------------
        # 3) Sort by NEW score (DESC best-first) and take top-k
        # -------------------------------
        ranked_desc = sorted(enriched, key=lambda r: r["score"], reverse=True)
        top_paths = ranked_desc[:k]

        # -------------------------------
        # 4) Rank-weighted FIRST-action voting (UNCHANGED)
        # -------------------------------
        actions = ("left", "right", "forward")
        scores = {a: 0.0 for a in actions}
        contrib = {a: [] for a in actions}

        def rank_weight(i: int) -> float:
            if weight == "harmonic":
                return 1.0 / (i + 1)        # 1, 1/2, 1/3, ...
            if weight == "exp":
                return float(alpha ** i)    # 1, α, α^2, ...
            return float(max(k - i, 1))     # linear: k, k-1, ..., 1

        for i, rec in enumerate(top_paths):
            seq = rec.get("seq", [])
            if len(seq) < require_min_depth:
                continue
            a0 = seq[0]
            if a0 not in scores:
                continue
            w = rank_weight(i)
            scores[a0] += w
            # keep debug consistent but show S=score instead of the builder's gain
            contrib[a0].append((i + 1, w, rec.get("score", 0.0), list(seq)))

        def tiebreak(key):
            # Prefer higher vote; then fixed order forward > left > right
            order = {"forward": 2, "left": 1, "right": 0}
            return (key[1], order.get(key[0], -1))

        best_action = max(scores.items(), key=tiebreak)[0]

        # -------------------------------
        # 5) Verbose debug
        # -------------------------------
        if verbose:
            print("\n[TOP-K FIRST-ACTION VOTING]")
            print(f"  k={k} weight={weight} alpha={alpha} include_pruned={include_pruned}")
            for i, rec in enumerate(top_paths, 1):
                a0 = rec["seq"][0] if rec["seq"] else "∅"
                s_terms = rec.get("score_terms", {})
                print(f"  #{i:02d} S={rec['score']:.3f}  first={a0:<7} depth={rec.get('depth')} seq={rec.get('seq')} "
                    f"pruned={rec.get('pruned_reason')}")
                print(f"        └ new_gain={s_terms.get('new_gain', 0.0):.3f}  "
                    f"B*dist_start={s_terms.get('B',0.0)*s_terms.get('dist_from_start',0.0):.3f}  "
                    f"C*dist_graph={s_terms.get('C',0.0)*s_terms.get('dist_from_graph_nodes',0.0):.3f}")

            print("  Action scores:")
            for a in actions:
                print(f"    {a:<7} → {scores[a]:.3f}  via {len(contrib[a])} hits")
                for rank, w, S, s in contrib[a]:
                    print(f"       - rank#{rank:02d} w={w:.3f} S={S:.3f} seq={s}")

        # External stash for inspection
        self._last_top_paths = top_paths
        self._last_action_scores = scores

    return best_action, scores, top_paths """    