# dreamer_mg/world_model_utils.py
# ----------------------------------------------------------------------
#  ✧  World-model utilities for MiniGrid collision prediction & planning
# ----------------------------------------------------------------------
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
 

# ─────────────────────────────── constants ────────────────────────────
State    = namedtuple("State", ["x", "y", "d"])          # planner state
DIR_VECS = [(1,0), (0,1), (-1,0), (0,-1)]                # 0:right 1:down …

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

class WMPlanner:
    """High-level wrapper: loading, belief update, collision check, A* and render."""
    # ──────────────────────────────────────────────────────────────
    def __init__(self, ckpt: str, cfg: str | None = None, device="cpu"):
        self.wm = self.load_world_model2(ckpt_path=ckpt)
        self.wm.device = device
        self.device = device

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
    # ══════════════════════════════════════════════════════════════════════
    # 3.  A*  PLANNER  (astar_prims)  –  uses wm_predict_collision
    # ══════════════════════════════════════════════════════════════════════
    def heuristic(self,s: State, g: State) -> int:
        manh = abs(s.x - g.x) + abs(s.y - g.y)
        turn = min((s.d - g.d) % 4, (g.d - s.d) % 4)
        return manh + turn
    def astar_prims(self,wm,
                    belief_zd,                  # (z,d) at the *start* frame
                    start: State,
                    goal:  State,
                    max_actions: int = 50,      # ← NEW hard budget
                    num_rollouts: int = 8,
                    verbose: bool = False):

        # priority-queue items: (f, g, state, seq, belief)
        pq      = [(self.heuristic(start, goal), 0, start, [], belief_zd)]
        g_score = {start: 0}
        closed  = set()

        # best fallback if no exact path exists
        best_dist = float("inf")
        best_seq  = []

        #self.astar_prims._wall_id = 0          # debug image counter

        while pq:
            f, g, (x, y, d), seq, belief = heapq.heappop(pq)
            if (x, y, d) in closed:
                continue
            closed.add((x, y, d))

            # update fallback record
            dist_xy = math.hypot(x - goal.x, y - goal.y)
            if dist_xy < best_dist:
                best_dist, best_seq = dist_xy, seq

            if verbose:
                print(f"[A*] pop {State(x,y,d)}  g={g}  f={f}  seq={seq}")

            # ───── success ─────────────────────────────────────────────
            if (x, y, d) == (goal.x, goal.y, goal.d):
                return seq                                   # ✔ exact plan

            # budget check: already at limit → do not expand
            if g >= max_actions:
                continue

            # ───── expand successors ───────────────────────────────────
            for act in ("left", "right", "forward"):

                # pose after action
                if act == "forward":
                    dx, dy = DIR_VECS[d]
                    nx, ny, nd = x + dx, y + dy, d
                elif act == "left":
                    nx, ny, nd = x, y, (d - 1) % 4
                else:              # right
                    nx, ny, nd = x, y, (d + 1) % 4
                ns = State(nx, ny, nd)
                if ns in closed:
                    continue

                # belief one step ahead (no-grad)
                with torch.no_grad():
                    z, h = belief
                    onehot = F.one_hot(
                        torch.tensor([wm.idx[act]], device=wm.device),
                        num_classes=wm.action_size).float()  # (1,3)

                    h_next = wm.rssm.recurrent_model(z, onehot, h)
                    _, z_next = wm.rssm.transition_model(h_next)
                belief_next = (z_next.detach(), h_next.detach())

                # collision check only for forward
                if act == "forward":
                    if self.is_wall_ahead_now(wm, belief, debug=False):
                        if verbose:
                            print("   prune (wall ahead now)")
                        continue

                # push successor if still within action budget
                new_seq = seq + [act]
                g2 = g + 1
                if g2 > max_actions:
                    continue            # would exceed budget

                h2 = self.heuristic(ns, goal)
                f2 = g2 + h2
                if g2 < g_score.get(ns, float("inf")):
                    g_score[ns] = g2
                    heapq.heappush(pq, (f2, g2, ns, new_seq, belief_next))

            
        print(f"[A*] no path; returning closest seq (dist={best_dist:.2f})")
        return best_seq            # may be empty if start==goal & unreachable

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
    
    def update_belief_from_obs(self,obs_dict,belief_zd, prev_onehot):
            
        frame = obs_dict["image"].transpose(2,0,1) / 255.0-0.5   # ★ (3,64,64) float
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
# ──────────────────────────────────────────────────────────────────────
#  QUICK SELF-TEST  (python -m dreamer_mg.world_model_utils)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time, gym, gym_minigrid
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgActionObsWrapper
    from world_model_utils import DictResizeObs ,WMPlanner
    from torchvision.utils import save_image
    CKPT = "runs/mg_collision/20250704-220917/ckpt/iter00800.pt"
    planner = WMPlanner(CKPT, device=DEVICE)      # loads WM + sets device
    print("✓ world-model loaded")
    wm=planner.wm
    # ------------ env --------------------------------------------------
    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0",
                   rooms_in_row=3, rooms_in_col=4)
    env.seed(218)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env)
    env = DictResizeObs(env, (64, 64))

    # ------------ first observation → initial belief -------------------
    obs  = env.reset()
    frame = obs["image"].transpose(2,0,1)/255.0 - 0.5
    belief = planner.wm_update_belief(wm,prev_z_d=None,frame_rgb=frame,prev_action_onehot=None)
    obs, _, done, _ = env.step(env.actions.forward)   
    prev_act_1h = planner.onehot("forward", wm)
    belief=planner.update_belief_from_obs(obs,belief,prev_act_1h)
    # ------------ plan to a toy goal -----------------------------------
    start = State(*obs["pose"])        # current (x,y,d)
    print(start)
    goal  = State(4, 2, 0)             # arbitrary demo target

    actions = planner.astar_prims(wm,belief, start, goal, verbose=True)
    print("plan :", actions)

    # ------------ render imagined rollout ------------------------------
    frames = planner.render_plan(wm,belief, actions, include_last=True)
    Path("dbg").mkdir(exist_ok=True, parents=True)
    save_image(torch.stack(frames),
               "dbg/demo_rollout.png", nrow=len(frames),
               normalize=True, scale_each=False)
    print("saved dbg/demo_rollout.png")


'''    
# ══════════════════════════════════════════════════════════════════════
# 4.  QUICK SELF-TEST  (python -m dreamer_mg.world_model_utils)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time, gym, gym_minigrid
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgActionObsWrapper
    from world_model_utils import onehot, DictResizeObs         # already defined
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image
    import os, itertools
    from pathlib import Path
    # three relative target offsets (dx,dy,dir)
    LOCAL_GOALS = [(4, 2, 0), (-3, 0, 2), (6, 2, 3)]   # ahead / left / right

    def absolute_goal(env, dx, dy, dir_):
       return State(env.agent_pos[0] + dx,
                     env.agent_pos[1] + dy,
                     dir_)
        
        return State(dx,
                     dy,
                     dir_)

    def query_three_goals(step,start):
        print(f"\n══════════  QUERY @ step {step}  ══════════")
        for i, (dx,dy,dir_) in enumerate(LOCAL_GOALS, 1):
            g = absolute_goal(env, dx, dy, dir_)
            print(g)
            px, py, pd = start
            print("start",px,py,pd)
            with torch.inference_mode(): 
                plan = astar_prims(
                        wm, belief_zd,
                        start=State(int(px), int(py), int(pd)),
                        goal =g,
                        verbose=True    # full expansion prints
                    )
            status = "NO-PATH" if not plan else f"plan {plan}"
            print(f"[goal {i}]  {g}  →  {status}")
    CKPT = "runs/mg_collision/20250704-220917/ckpt/iter00800.pt"

    wm   = load_world_model2(CKPT)
    print(f"[demo] loaded WM from {CKPT}")
    print("onehot('forward'):", onehot("forward", wm))
    print("onehot mapping OK ✓")
    
    def debug_rng(env, label):
        print(f"[{label}] core RNG state hash:",
            hash(env.unwrapped.np_random.get_state()[1].tobytes()))
    # ------------ env --------------------------------------------------
    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0",
                   rooms_in_row=3, rooms_in_col=4)
    env.seed(218)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env)
    env = DictResizeObs(env, (64, 64))
    #env = ChannelFirstEnv(env)                # (3,64,64)
    
    K = 5                                    # how many real steps to collect
    frames, acts = [], []

    obs = env.reset()



    belief_zd      = None
    prev_act_1h = None
    belief_zd = update_belief_from_obs(obs,belief_zd, prev_act_1h)
    obs, _, done, _ = env.step(env.actions.forward)
    env.render(tile_size=64)

    # ------------ belief & bookkeeping --------------------------------
    prev_act_1h = onehot("forward", wm)
    FPS            = 6
    
    # ------------------------------------------------------------------
    belief_zd = update_belief_from_obs(obs,belief_zd, prev_act_1h)
    query_three_goals(step=0,start=obs['pose'])
    
    
    save_decoder(belief_zd,obs)
    probe_decoder(wm, belief_zd, obs,     # current posterior
              action_plan = ["forward","forward","left","forward"])
    
    # ------------ phase 1 : drive 4 steps straight --------------------
    for t in range(4):
        belief_zd = update_belief_from_obs(obs,belief_zd, prev_act_1h)
        prev_act_1h = onehot("forward", wm)          # hard-coded forward
        obs, _, done, _ = env.step(env.actions.forward)
        #print(obs['pose'])
        env.render(tile_size=64)
        time.sleep(1.0 / FPS)
    

    query_three_goals(step="4+",start=obs['pose'])
    
    save_decoder(belief_zd,obs)
    probe_decoder(wm, belief_zd, obs,     # current posterior
              action_plan = ["forward","forward","left","forward"])
    
    #z0,d0=belief_zd
    
    # ------------ phase 2 : drive 2 more steps ------------------------
    for t in range(2):
        belief_zd = update_belief_from_obs(obs,belief_zd, prev_act_1h)
        prev_act_1h = onehot("forward", wm)
        obs, _, done, _ = env.step(env.actions.forward)
        env.render(tile_size=64)
        time.sleep(1.0 / FPS)
    
    save_decoder(belief_zd,obs)

    query_three_goals(step="6+",start=obs['pose'])
    probe_decoder(wm, belief_zd, obs,     # current posterior
              action_plan = ["forward","forward","left","forward"])
    env.close()
 




 def wm_predict_collision_from_belief(
            self,wm,
            belief_zd: tuple[torch.Tensor, torch.Tensor],
            seq: list[str],
            num_rollouts: int =2 ,
            thresh: float = 0.01,           # reward < thresh ⇒ “collision”
            debug: bool = True,            # ⬅️  turn prints on/off
            _save_id: list[int] = [0]
    ) -> bool:
        """
        Return True iff *any* rollout predicts a reward below `thresh`
        on *any* step of `seq`.  Uses the checkpoint’s reward head.
        """

        z0, d0 = (x.detach() for x in belief_zd)         # (1,Z) (1,H)
        action_size = wm.action_size
        idx   = torch.tensor([wm.idx[a] for a in seq], device=wm.device)
        onehs = F.one_hot(idx, num_classes=action_size).float()   # (T,action_size)

        if debug:
            print("\n[COLL-CHK] -------------------------------------------------")
            print(" sequence :", seq)
            print(" thresh   :", thresh, "| rollouts:", num_rollouts)
            print(" start z‖d:", tuple(x.shape for x in belief_zd))

        unsafe = False
        rollout_counter = 0
        for r in range(num_rollouts):
            z, d = z0.clone(), d0.clone()
            frames_dbg, rewards_dbg = [], []
            if debug:  # decode starting frame as well (posterior @ T)
                frames_dbg.append(wm.decoder(z, d).mean.squeeze(0).cpu())
                rewards_dbg.append(float("nan"))         # no reward yet

            for t, a in enumerate(onehs, 1):
                d = wm.rssm.recurrent_model(z, a.unsqueeze(0), d)
                _, z = wm.rssm.transition_model(d)

                r_pred = wm.reward_predictor(z, d).mean.item()
                if debug:
                    print(f"      step {t:<2} act={seq[t-1]:>7}  "
                        f"reward_pred={r_pred:+.3f}")
                    frames_dbg.append(wm.decoder(z, d).mean.squeeze(0).cpu())
                    rewards_dbg.append(r_pred)

                if r_pred < thresh:
                    unsafe = True
                    break

            if debug:
                tag = f"{_save_id[0]:05d}"
                _save_id[0] += 1
                self._save_rollout_grid(frames_dbg, rewards_dbg,
                                tag)
                

            if unsafe:
                break
                


    for k in range(K):
        img = obs["image"].astype(np.float32).transpose(2,0,1) / 255.0 - 0.5
        frames.append(torch.tensor(img))

        act_name = "forward"                 # pick any pattern you like
        acts.append(onehot(act_name,wm))
        obs, _, _, _ = env.step(getattr(env.actions, act_name))
    frames = torch.stack(frames, dim=0)      # (K,3,64,64)
    acts   = torch.stack(acts,   dim=0)      # (K, action_size)
    frames_seq = frames.unsqueeze(0).to(DEVICE)   # (1,K,3,64,64)
    acts_seq   = acts  .unsqueeze(0).to(DEVICE)   # (1,K,action_size)


    emb_seq    = wm.encoder(frames_seq)                  # (1,K,1024)
        
    # 1. initial deterministic & prior *before* seeing first obs
    prior_z, det = wm.rssm.recurrent_model_input_init(1)

    # 2. **posterior at t = 0**  (uses first embedding)
    post_dist, post_z = wm.rssm.representation_model(emb_seq[:, 0], det)
    lat_posts  = [post_z]           # list of (1,Z)
    lat_dets   = [det]              # list of (1,H)

    # 3. roll through time 1 … K-1  (exactly like training)
    for t in range(1, K):
        det = wm.rssm.recurrent_model(post_z, acts_seq[:, t-1], det)
        _,  prior_z = wm.rssm.transition_model(det)
        _,  post_z  = wm.rssm.representation_model(emb_seq[:, t], det)

        lat_posts.append(post_z)                 # posterior z_t
        lat_dets .append(det)                    # deterministic h_t

    # 4. stack to tensors (1,K,Z) / (1,K,H)
    lat_posts = torch.stack(lat_posts, dim=1)
    lat_dets  = torch.stack(lat_dets , dim=1)

    # 5. reconstruction of EVERY real frame
    recon = wm.decoder(lat_posts, lat_dets).mean.cpu()  # (1,K,3,64,64)
    grid = torch.cat([                   # ② match lengths
            frames[1:].cpu(),            # o₁ … o_K₋₁
            recon.squeeze(0)[1:]         # r₁ … r_K₋₁
            ], dim=0)                   # (2·(K-1),3,64,64)
    
    import torchvision.utils as vutils, os
    outdir = Path("dbg"); outdir.mkdir(exist_ok=True)

    # ----- Grid 1 : GT @ T  |  posterior @ T  |  prior @ T -------------
    T = K - 1
    prior_img = wm.decoder(lat_posts[:, T], lat_dets[:, T]).mean.squeeze(0).cpu()
    post_img  = recon.squeeze(0)[T]

    grid1 = torch.stack([frames[T],         # GT
                        post_img,          # posterior recon
                        prior_img])        # prior recon
    vutils.save_image(grid1, outdir/"recon_last.png", nrow=3, normalize=True)

    # ----- Grid 2 : GT @ T  +  imagined rollout ------------------------
    cmd = ["right", "left", "left", "forward"]
    act_idx  = torch.tensor([wm.idx[a] for a in cmd], device=DEVICE)
    dream_z  = lat_posts[:, T]               # start from *posterior* z_T
    dream_d  = lat_dets[:,  T]
    dreams   = []
    for i in act_idx:
        one_hot = F.one_hot(i, num_classes=wm.action_size).float().unsqueeze(0)
        dream_d = wm.rssm.recurrent_model(dream_z, one_hot, dream_d)
        _, dream_z = wm.rssm.transition_model(dream_d)
        dreams.append(wm.decoder(dream_z, dream_d).mean.squeeze(0).cpu())

    grid2 = torch.stack([frames[T]] + dreams)
    vutils.save_image(grid2, outdir/"dream_rollout.png",
                    nrow=len(cmd)+1, normalize=True)
    print("wrote dbg/recon_last.png and dbg/dream_rollout.png")


def astar_prims(wm,
                belief_zd,              # (z,d) of the *start* observation
                start: State,
                goal:  State,
                verbose: bool = False):

    pq      = [(heuristic(start, goal), 0, start, [], belief_zd)]
    g_score = {start: 0}
    closed  = set()
    astar_prims._wall_id = 0
    while pq:
        f, g, (x,y,d), seq, belief = heapq.heappop(pq)
        if (x,y,d) in closed:           # state closed, not the path
            continue
        closed.add((x,y,d))
        if verbose:
            print(f"[A*] pop {State(x,y,d)} g={g} f={f} seq={seq}")

        if (x,y,d) == (goal.x, goal.y, goal.d):
            return seq                  # ✔ found plan

        for act in ("left","right","forward"):

            # --- successor pose --------------------------------------
            if act == "forward":
                dx, dy = DIR_VECS[d];  nx, ny, nd = x+dx, y+dy, d
            elif act == "left":
                nx, ny, nd = x, y, (d-1) % 4
            else:   # right
                nx, ny, nd = x, y, (d+1) % 4
            ns = State(nx, ny, nd)
            if ns in closed:
                continue

            # --- advance belief *one* step ---------------------------
            z, h = belief                       # each (1,dim)
            onehot = torch.nn.functional.one_hot(
                        torch.tensor([wm.idx[act]], device=wm.device),
                        num_classes=wm.action_size).float()       # (1,3)

            # deterministic update first
            h_next = wm.rssm.recurrent_model(z, onehot, h)
            # prior latent (no image available)
            _, z_next = wm.rssm.transition_model(h_next)
            belief_next = (z_next.detach(), h_next.detach())

            # --- collision if that **single** step is forward --------
            if act == "forward":
                wall_now = is_wall_ahead_now(
                            wm, belief,                # current belief
                            debug=False,               # set True to save PNGs
                            tag=f"wall{astar_prims._wall_id:05d}")
                astar_prims._wall_id += int(wall_now)

                if wall_now:
                    if verbose: print("   prune (wall ahead now)")
                    continue

            # --- push successor -------------------------------------
            new_seq = seq + [act]
            g2, h2 = g + 1, heuristic(ns, goal)
            f2     = g2 + h2
            if g2 < g_score.get(ns, float('inf')):
                g_score[ns] = g2
                heapq.heappush(pq, (f2, g2, ns, new_seq, belief_next))

    return []     # no safe path

def astar_prims(wm,
                belief_zd: tuple[torch.Tensor, torch.Tensor],
                start: State,
                goal:  State,
                num_rollouts: int = 8,
                verbose: bool = False) -> list[str]:
    """
    A* in (x,y,dir) space.  Any forward-move whose *entire prefix* is
    judged unsafe by the world-model is pruned away.
    """
    pq      = [(heuristic(start, goal), 0, start, [])]      # (f,g,s,seq)
    g_score = {start: 0}
    closed  = set()

    while pq:
        f, g, (x,y,d), seq = heapq.heappop(pq)
        if (x,y,d) in closed:
            continue
        closed.add((x,y,d))
        if verbose:
            print(f"[A*] pop  {State(x,y,d)}  g={g} f={f}  seq={seq}")

        if (x,y,d) == (goal.x, goal.y, goal.d):
            return seq                                    # success ✔

        for act in ("left", "right", "forward"):
            if act == "forward":
                dx, dy = DIR_VECS[d];  nx, ny, nd = x+dx, y+dy, d
            elif act == "left":
                nx, ny, nd = x, y, (d-1) % 4
            else:
                nx, ny, nd = x, y, (d+1) % 4

            ns = State(nx, ny, nd)
            if ns in closed:
                continue

            new_seq = seq + [act]

            # ── collision check *only* if the step is a forward move ──
            if act == "forward":
                unsafe = wm_collision_by_decoder(
                            wm, belief_zd, new_seq,
                            num_rollouts=num_rollouts,
                            debug=False)
                if unsafe:
                    if verbose:
                        print("   prune (predicted collision)",new_seq)
                    continue

            # ── push successor on the heap ────────────────────────────

            g2, h2 = g+1, heuristic(ns, goal)
            f2     = g2 + h2
            if g2 < g_score.get(ns, float("inf")):
                g_score[ns] = g2
                heapq.heappush(pq, (f2, g2, ns, new_seq))

    return []                                               # no safe path found

def step_latent_batch(wm, z_batch, d_batch, act_idx):
    """Advance every sample in the batch by ONE imagined step."""
    R = z_batch.size(0)
    a1h = F.one_hot(act_idx, num_classes=wm.action_size) \
            .float().unsqueeze(0).expand(R, -1)          # (R,action)
    d_batch = wm.rssm.recurrent_model(z_batch, a1h, d_batch)
    _, z_batch = wm.rssm.transition_model(d_batch)
    return z_batch, d_batch
# ───────────────────────────────────────────────────────────── structs
MCNode = namedtuple("MCNode",
    "g x y d z dvec z_s d_s seq")                      # f kept outside heap key  # keep f outside (used only in heap key)

# ───────────────────────────────────────────────────────────── search
@torch.no_grad()
def astar_prims(
        wm, belief_zd, start, goal,
        R=8, sigma=0.01, thresh=-0.35, K=1,
        verbose=False):
    """
    A* with batched Monte-Carlo collision pruning.
    R       : # latent samples per node
    sigma   : Gaussian noise std-dev on z
    thresh  : reward < thresh ⇒ collision
    K       : prune if ≥K of R samples collide
    """

    # ---------- start node ---------------------------------------------------
    z0, d0 = (t.detach() for t in belief_zd)          # (1,Z) (1,H)
    #z_samp = z0.repeat(R, 1) + sigma*torch.randn(R, z0.size(1), device=wm.device)
    #d_samp = d0.repeat(R, 1)
    z_samp = z0.repeat(R, 1)          # no additive noise
    d_samp = d0.repeat(R, 1)
    start_node = MCNode(0, start.x, start.y, start.d,
                        z0, d0, z_samp, d_samp, [])

    pq = []
    counter = itertools.count()
    heapq.heappush(pq, (heuristic(start, goal), next(counter), start_node))

    closed = set()

    while pq:
        _, _, node = heapq.heappop(pq)
        key = (node.x, node.y, node.d)
        if key in closed:
            continue
        closed.add(key)

        if key == (goal.x, goal.y, goal.d):
            return node.seq                                # ✔ path

        for act in ("left", "right", "forward"):
            # ---- local motion ------------------------------------------------
            if act == "forward":
                dx, dy = DIR_VECS[node.d]
                nx, ny, nd = node.x + dx, node.y + dy, node.d
            elif act == "left":
                nx, ny, nd = node.x, node.y, (node.d - 1) % 4
            else:  # right
                nx, ny, nd = node.x, node.y, (node.d + 1) % 4

            if (nx, ny, nd) in closed:
                continue

            act_idx = torch.tensor(wm.idx[act], device=wm.device)

            # ---- advance all R samples ONE step -----------------------------
            z1_s, d1_s = step_latent_batch(wm, node.z_s, node.d_s, act_idx)

            # ---- collision test ONLY for forward ----------------------------
            if act == "forward":
                print("we entered forward")
                r_preds = wm.reward_predictor(z1_s, d1_s).mean   # (R,)
                print("r_preds",r_preds, r_preds < thresh, thresh )
                if (r_preds < thresh).sum().item() >= K:
                    if verbose:
                        print(f" prune forward  reward min={r_preds.min():+.3f}")
                    continue

            # ---- push successor --------------------------------------------
            z1_det, d1_det = z1_s[0:1], d1_s[0:1]      # determinist. rep.
            seq   = node.seq + [act]
            g2    = node.g + 1
            f2    = g2 + heuristic(State(nx, ny, nd), goal)

            new_node = MCNode(g2, nx, ny, nd,
                              z1_det, d1_det, z1_s, d1_s, seq)
            heapq.heappush(pq, (f2, next(counter), new_node))

    return []                                           # no safe path

Node = namedtuple("Node", "g x y d z dvec seq")        # f kept outside heap key

@torch.no_grad()
def step_one(wm, z, d, act_idx):
    """ONE deterministic RSSM prior step (no observation)."""
    a1h = F.one_hot(act_idx, num_classes=wm.action_size).float().unsqueeze(0)
    d   = wm.rssm.recurrent_model(z, a1h, d)
    _, z = wm.rssm.transition_model(d)
    return z, d

@torch.no_grad()
def astar_prims(
        wm, belief_zd, start, goal,
        thresh=0.033, verbose=False):
    """
    A* where only 'forward' moves are run through reward_predictor;
    left/right are always allowed.  One RSSM step per edge.
    """

    z0, d0 = (t.detach() for t in belief_zd)
    start_n = Node(0, start.x, start.y, start.d, z0, d0, [])

    pq      = []
    counter = itertools.count()
    heapq.heappush(pq, (heuristic(start, goal), next(counter), start_n))
    closed  = set()

    while pq:
        _, _, node = heapq.heappop(pq)
        key = (node.x, node.y, node.d)
        if key in closed:
            continue
        closed.add(key)

        if key == (goal.x, goal.y, goal.d):
            return node.seq                               # ✔ path

        for act in ("left", "right", "forward"):
            # ----- next grid state -----------------------------------
            if act == "forward":
                dx, dy = DIR_VECS[node.d]
                nx, ny, nd = node.x + dx, node.y + dy, node.d
            elif act == "left":
                nx, ny, nd = node.x, node.y, (node.d - 1) % 4
            else:
                nx, ny, nd = node.x, node.y, (node.d + 1) % 4

            if (nx, ny, nd) in closed:
                continue

            act_idx = torch.tensor(wm.idx[act], device=wm.device)

            # ----- advance ONE latent step ---------------------------
            z1, d1 = step_one(wm, node.z, node.dvec, act_idx)

            # ----- collision check ONLY for forward ------------------
            if act == "forward":
                r_pred = wm.reward_predictor(z1, d1).mean.item()
                if verbose:
                    print(f"   {node.seq + [act]}  reward={r_pred:+.3f}")
                if r_pred > thresh:
                    if verbose:
                        print("     prune (reward below thresh)")
                    continue

            # ----- push successor -----------------------------------
            g2 = node.g + 1
            f2 = g2 + heuristic(State(nx, ny, nd), goal)
            new_node = Node(g2, nx, ny, nd, z1, d1, node.seq + [act])
            heapq.heappush(pq, (f2, next(counter), new_node))

    return []                                            # no safe path''' 