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
from attrdict import AttrDict 
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym.wrappers import ResizeObservation
from dreamer.envs.wrappers import ChannelFirstEnv
import matplotlib.pyplot as plt

# ─────────────────────────────── constants ────────────────────────────
State    = namedtuple("State", ["x", "y", "d"])          # planner state
DIR_VECS = [(1,0), (0,1), (-1,0), (0,-1)]                # 0:right 1:down …

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════
# 1.  LOADING  (one-liner:  wm = load_world_model("…/iter0500.pt") )
# ══════════════════════════════════════════════════════════════════════

def load_world_model2(ckpt_path: str,
                     config_yaml: str = "dreamer/configs/minigrid-default-temp.yml"):
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

    # ---------- 1. discover sizes from weight shapes -------------------
    Wrec = ckpt["modules"]["rssm"]["recurrent_model.linear.weight"]      # (H, in)
    H, in_dim        = Wrec.shape
    Wtrans = ckpt["modules"]["rssm"]["transition_model.network.2.weight"]# (2*Z, H)
    stoch_sz         = Wtrans.shape[0] // 2
    act_size         = in_dim - stoch_sz            # the checkpoint’s action space

    # ---------- 2. clone & patch the yaml ------------------------------
    
    cfg = yaml.safe_load(open(config_yaml))
    cfg = copy.deepcopy(cfg)                         # keep original intact

    cfg = AttrDict(cfg)  
    cfg['parameters']['dreamer']['deterministic_size'] = H
    cfg['parameters']['dreamer']['stochastic_size']    = stoch_sz

    enc  = Encoder((3, 64, 64), cfg).to(DEVICE)
    dec = Decoder((3, 64, 64), cfg).to(DEVICE)
    rssm = RSSM(action_size=act_size, config=cfg).to(DEVICE)

    enc .load_state_dict(ckpt["modules"]["encoder"]);  enc .eval()
    rssm.load_state_dict(ckpt["modules"]["rssm"   ]);  rssm.eval()
    dec.load_state_dict(ckpt["modules"]["decoder"]); dec.eval()

    wm = lambda: None
    wm.encoder = enc
    wm.rssm    = rssm
    wm.decoder = dec
    wm.action_size = act_size                        # handy later
    wm.idx = {                           #  ← canonical MiniGrid ids
        "left":   0,
        "right":  1,
        "forward":2,
        "pickup": 3,
        "drop":   4,
        "toggle": 5,
        "done":   6,    }
    print("rssm expects action dim:", wm.action_size)
    return wm
def onehot(action_name: str, wm):
    v = torch.zeros(wm.action_size, device=DEVICE)
    idx = wm.idx.get(action_name, None)
    if idx is not None:
        v[idx] = 1.
    return v
# ══════════════════════════════════════════════════════════════════════
# 2.  COLLISION PREDICTION  (wm_predict_collision)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
@torch.no_grad()
def wm_update_belief(wm,
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
        zprev, dprev = wm.rssm.recurrent_model_input_init(1)
    else:
        zprev, dprev = prev_z_d

    # recurrent model if we have an *action that already happened*
    if zprev is not None and prev_action_onehot is not None:
        dprev = wm.rssm.recurrent_model(zprev,
                                        prev_action_onehot.unsqueeze(0),  # (1,7)
                                        dprev)

    # encode observation & run representation model
    img = torch.tensor(frame_rgb, dtype=torch.float32,
                       device=DEVICE).unsqueeze(0)
    emb = wm.encoder(img).view(1, -1)
    _, zt = wm.rssm.representation_model(emb, dprev)   # posterior

    return zt.detach(), dprev.detach()


@torch.no_grad()
def wm_predict_collision_from_belief(wm,
                                     belief_zd: tuple[torch.Tensor,torch.Tensor],
                                     action_seq: list[str],
                                     threshold: float = .5,
                                     num_rollouts: int = 8) -> bool:
    """
    Same semantics as the stateless version but *starts* from an already
    updated belief (z_t,d_t).  No RGB frame is needed here.
    """
    z0, d0 = belief_zd
    table  = {'forward':[1,0,0], 'right':[0,1,0], 'left':[0,0,1]}
    acts = torch.stack([onehot(a, wm) for a in action_seq]).unsqueeze(0)

    hits = 0
    for _ in range(num_rollouts):
        z, d = z0, d0
        for t, a in enumerate(action_seq):
            d = wm.rssm.recurrent_model(z, acts[:, t], d)
            _, z = wm.rssm.transition_model(d)

            # ***replace this stub by your collision-head later***
            '''if a == "forward" and random.random() < 0.15:
                print("fired")
                hits += 1;  break'''

    return (hits / num_rollouts) >= threshold
# ══════════════════════════════════════════════════════════════════════
# 3.  A*  PLANNER  (astar_prims)  –  uses wm_predict_collision
# ══════════════════════════════════════════════════════════════════════
def heuristic(s: State, g: State) -> int:
    manh = abs(s.x - g.x) + abs(s.y - g.y)
    turn = min((s.d - g.d) % 4, (g.d - s.d) % 4)
    return manh + turn

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

            # collision check **from current belief**
            unsafe = wm_predict_collision_from_belief(
                        wm, belief_zd, new_seq,
                        num_rollouts=num_rollouts
                     )
            if unsafe:
                if verbose: print("   prune (predicted collision)")
                continue

            g2, h2 = g+1, heuristic(ns, goal)
            f2     = g2 + h2
            if g2 < g_score.get(ns, float("inf")):
                g_score[ns] = g2
                heapq.heappush(pq, (f2, g2, ns, new_seq))

    return []                                               # no safe path found
import cv2, gym
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

    CKPT = "runs/mg_collision/20250630-165944/ckpt/iter01210.pt"
    wm   = load_world_model2(CKPT)
    print(f"[demo] loaded WM from {CKPT}")
    

    # ------------ env --------------------------------------------------
    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0",
                   rooms_in_row=3, rooms_in_col=4)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env)
    env = DictResizeObs(env, (64, 64))
    #env = ChannelFirstEnv(env)                # (3,64,64)
    seed=218
    env.seed(seed)
    obs  = env.reset()
    obs, _, done, _ = env.step(env.actions.forward)
    env.render(tile_size=64)

    # ------------ belief & bookkeeping --------------------------------
    belief_zd      = None
    prev_act_1h    = None
    FPS            = 6
    # three relative target offsets (dx,dy,dir)
    LOCAL_GOALS = [(4, 2, 0), (-3, 0, 2), (6, 2, 3)]   # ahead / left / right

    def absolute_goal(env, dx, dy, dir_):
        '''return State(env.agent_pos[0] + dx,
                     env.agent_pos[1] + dy,
                     dir_)
        '''
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
            plan = astar_prims(
                     wm, belief_zd,
                     start=State(int(px), int(py), int(pd)),
                     goal =g,
                     num_rollouts=12,
                     verbose=False    # full expansion prints
                   )
            status = "NO-PATH" if not plan else f"plan {plan}"
            print(f"[goal {i}]  {g}  →  {status}")

    def update_belief_from_obs(obs_dict, prev_onehot):
        frame = obs_dict["image"].transpose(2,0,1) / 255.0   # ★ (3,64,64) float
        return wm_update_belief(wm, belief_zd, frame, prev_onehot)
    def probe_decoder(wm, belief_zd, obs, action_plan, outdir="recon_demo"):
        outdir = Path(outdir); outdir.mkdir(exist_ok=True)

        # ---------- 0. utilities --------------------------------
        def to_onehot(a_name):
            v = torch.zeros(wm.action_size, device=DEVICE)
            v[wm.idx[a_name]] = 1.0
            return v

        # ---------- 1. current posterior ------------------------
        rgb = obs["image"].astype(np.float32).transpose(2,0,1)/255.0
        z, d = belief_zd            # posterior of current frame (already updated)
        recon0 = wm.decoder(z, d).mean.squeeze(0).cpu()

        # ---------- 2. imagine K steps (priors) -----------------
        zs, ds, frames = [z], [d], [recon0]
        onehots = torch.stack([to_onehot(a) for a in action_plan]).unsqueeze(0)

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
        grid = torch.nn.functional.interpolate(grid, size=256,
                                            mode="nearest")  # pixel-art
        fname = outdir / f"probe_{len(list(outdir.glob('probe_*.png'))):03d}.png"
        save_image(grid, fname, nrow=len(frames)+1, normalize=True)
        print("saved", fname)
    
    def save_decoder(belief_zd,obs):
            # --- grab one RGB frame from the env -------------------------------
        OUTDIR   = Path("recon_demo")         # ./recon_demo/…
        OUTDIR.mkdir(exist_ok=True)

        z0,d0= wm.rssm.recurrent_model_input_init(1)                         # posterior (current belief)
        #z0,d0=belief_zd
        # ---- choose any action pattern you want the model to fantasise ----
        plan      = ["forward","left", "right", "forward", "forward"]   # length = K
        onehots   = torch.stack([onehot(a, wm) for a in plan]).unsqueeze(0)
        zs, ds    = [z0], [d0]

        # ---- latent roll-out (PRIOR predictions!) -------------------------
        for t in range(len(plan)):
            d_next  = wm.rssm.recurrent_model(zs[-1], onehots[:, t], ds[-1])
            _, z_next = wm.rssm.transition_model(d_next)      # sample from p(zₜ₊₁)
            zs.append(z_next);  ds.append(d_next)

        # ---- decode all latents ------------------------------------------
        with torch.no_grad():
            recons = torch.stack([wm.decoder(z, d).mean.squeeze(0).cpu()
                                for z, d in zip(zs, ds)])   # (K+1,3,64,64)

        # ---- build a pretty  grid  (1×truth  +  K+1×pred) -----------------
        truth = obs["image"].astype(np.float32).transpose(2,0,1)/255.0
        grid  = torch.cat([torch.tensor(truth).unsqueeze(0), recons], dim=0)

        # upscale to 256 px per tile for readability
        grid_big = torch.nn.functional.interpolate(grid, size=256,
                                                mode="bilinear", align_corners=False)

        fname = OUTDIR / f"recon_{len(list(OUTDIR.glob('recon_*.png'))):03d}.png"
        save_image(grid_big, fname, nrow=len(plan)+1, normalize=True)
        print(f"[demo] saved  →  {fname}")
        

    

    # ------------------------------------------------------------------
    belief_zd = update_belief_from_obs(obs, prev_act_1h)
    query_three_goals(step=0,start=obs['pose'])
    
    
    save_decoder(belief_zd,obs)
    probe_decoder(wm, belief_zd, obs,     # current posterior
              action_plan = ["forward","forward","left","forward"])
    z0,d0= belief_zd
    # after one real step -------------------------------------------------
    frame1 = obs["image"].astype(np.float32).transpose(2,0,1) / 255.0
    onehot_fwd = onehot("forward", wm).unsqueeze(0)            # (1,7)

    # 1️⃣ get posterior z₁,d₁ from the *real* next frame
    emb = wm.encoder(torch.tensor(frame1, device=DEVICE).unsqueeze(0))
    d0  = wm.rssm.recurrent_model(z0, onehot_fwd, d0)          # update hidden state
    _, z1 = wm.rssm.representation_model(emb.view(1, -1), d0)  # posterior

    # 2️⃣ decode that posterior              → how well do we reconstruct?
    recon_post = wm.decoder(z1, d0).mean.squeeze(0)     # (3,64,64)

    # 3️⃣ now imagine two *prior* steps ahead with the same “forward” action
    z, d = z1, d0
    priors = []
    for _ in range(2):
        d = wm.rssm.recurrent_model(z, onehot_fwd, d)
        _, z = wm.rssm.transition_model(d)     # PRIOR
        priors.append(wm.decoder(z, d).mean.squeeze(0).cpu())

    save_image(
        torch.stack([torch.tensor(frame1), recon_post] + priors),
        "recon_debug.png", nrow=5, normalize=True
    )
    print("wrote recon_debug.png")
    # ------------ phase 1 : drive 4 steps straight --------------------
    for t in range(4):
        belief_zd = update_belief_from_obs(obs, prev_act_1h)
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
        belief_zd = update_belief_from_obs(obs, prev_act_1h)
        prev_act_1h = onehot("forward", wm)
        obs, _, done, _ = env.step(env.actions.forward)
        env.render(tile_size=64)
        time.sleep(1.0 / FPS)
    
    save_decoder(belief_zd,obs)

    query_three_goals(step="6+",start=obs['pose'])
    probe_decoder(wm, belief_zd, obs,     # current posterior
              action_plan = ["forward","forward","left","forward"])
    env.close()
