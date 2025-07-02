# Fixed decoder testing functions for world_model_utils.py
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
def save_decoder_fixed(wm, belief_zd, obs, step_name=""):
    """
    Fixed version that properly uses current belief and handles tensor formats
    """
    from pathlib import Path
    from torchvision.utils import save_image
    import torch
    import numpy as np
    
    OUTDIR = Path("recon_demo")
    OUTDIR.mkdir(exist_ok=True)
    
    # Get current belief state (this is the key fix!)
    z_current, d_current = belief_zd
    
    # Plan for imagination
    plan = ["forward", "left", "right", "forward", "forward"]
    onehots = torch.stack([onehot(a, wm) for a in plan]).unsqueeze(0)  # (1, K, action_dim)
    
    # Start imagination from CURRENT belief, not initial state
    zs, ds = [z_current], [d_current]
    
    # Roll out in imagination (prior predictions)
    with torch.no_grad():
        for t in range(len(plan)):
            d_next = wm.rssm.recurrent_model(zs[-1], onehots[:, t], ds[-1])
            _, z_next = wm.rssm.transition_model(d_next)  # sample from prior
            zs.append(z_next)
            ds.append(d_next)
    
    # Decode all states (current + imagined future)
    recons = []
    with torch.no_grad():
        for z, d in zip(zs, ds):
            # Decoder expects (batch, z_dim) and (batch, d_dim)
            recon_dist = wm.decoder(z, d)
            recon_img = recon_dist.mean.squeeze(0).cpu()  # (3, 64, 64)
            recon_img = torch.clamp(recon_img, 0, 1)  # Ensure valid range
            recons.append(recon_img)
    
    # Get current observation as tensor
    truth = obs["image"].astype(np.float32).transpose(2,0,1) / 255.0  # (3,64,64)
    truth_tensor = torch.tensor(truth)
    
    # Combine truth + reconstructions
    all_images = [truth_tensor] + recons
    grid = torch.stack(all_images, dim=0)  # (K+2, 3, 64, 64)
    
    # Upscale for better visibility
    grid_big = torch.nn.functional.interpolate(
        grid, size=256, mode="bilinear", align_corners=False
    )
    
    # Save with step identifier
    fname = OUTDIR / f"recon_{step_name}_{len(list(OUTDIR.glob('recon_*.png'))):03d}.png"
    save_image(grid_big, fname, nrow=len(plan)+2, normalize=False)
    print(f"[demo] saved reconstruction → {fname}")

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

def test_current_reconstruction(wm, belief_zd, obs):
    """
    Test how well the current belief reconstructs the current observation
    """
    import torch
    from torchvision.utils import save_image
    
    z_current, d_current = belief_zd
    
    with torch.no_grad():
        # Reconstruct current state
        recon_dist = wm.decoder(z_current, d_current)
        recon_img = recon_dist.mean.squeeze(0).cpu()  # (3, 64, 64)
        
        
        # Get ground truth
        truth = obs["image"].astype(np.float32).transpose(2,0,1) / 255.0
        truth_tensor = torch.tensor(truth)
        
        # Side by side comparison
        comparison = torch.stack([truth_tensor, recon_img], dim=0)
        comparison_big = torch.nn.functional.interpolate(
            comparison, size=256, mode="bilinear", align_corners=False
        )
        
        save_image(comparison_big, "current_recon_test.png", nrow=2, normalize=False)
        print("Saved current reconstruction test → current_recon_test.png")
        
        # Print reconstruction quality metrics
        mse = torch.nn.functional.mse_loss(recon_img, truth_tensor)
        print(f"Reconstruction MSE: {mse.item():.6f}")
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


def step_by_step_reconstruction_test(wm, env, num_steps=5):
    """
    Complete test that steps through environment and shows reconstructions
    """
    import time
    from pathlib import Path
    from torchvision.utils import save_image
    import torch
    
    OUTDIR = Path("step_by_step_recon")
    OUTDIR.mkdir(exist_ok=True)
    
    # Reset environment
    obs = env.reset()
    belief_zd = None
    prev_act_1h = None
    
    all_comparisons = []
    
    for step in range(num_steps):
        print(f"\n=== Step {step} ===")
        
        # Update belief from observation
        belief_zd = wm_update_belief(wm, belief_zd, 
                                   obs["image"].transpose(2,0,1) / 255.0, 
                                   prev_act_1h)
        
        # Test current reconstruction
        z_current, d_current = belief_zd
        with torch.no_grad():
            recon_dist = wm.decoder(z_current, d_current)
            recon_img = recon_dist.mean.squeeze(0).cpu()
            
            
            # Ground truth
            truth = obs["image"].astype(np.float32).transpose(2,0,1) / 255.0
            truth_tensor = torch.tensor(truth)
            
            # Store comparison
            comparison = torch.stack([truth_tensor, recon_img], dim=0)
            all_comparisons.append(comparison)
            
            # Calculate and print metrics
            mse = torch.nn.functional.mse_loss(recon_img, truth_tensor)
            print(f"Step {step} Reconstruction MSE: {mse.item():.6f}")
        
        # Take action and get next observation
        action = env.actions.forward  # or choose randomly
        prev_act_1h = onehot("forward", wm)
        obs, _, done, _ = env.step(action)
        
        if done:
            break
    
    # Save all comparisons as a grid
    if all_comparisons:
        # Stack all comparisons vertically
        full_grid = torch.cat(all_comparisons, dim=0)  # (2*num_steps, 3, 64, 64)
        full_grid_big = torch.nn.functional.interpolate(
            full_grid, size=256, mode="bilinear", align_corners=False
        )
        
        save_image(full_grid_big, OUTDIR / "full_sequence.png", 
                  nrow=2, normalize=False)
        print(f"Saved full sequence → {OUTDIR / 'full_sequence.png'}")


# Updated main function with fixed decoder testing
if __name__ == "__main__":
    import time, gym, gym_minigrid
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgActionObsWrapper
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image
    import os, itertools
    from pathlib import Path
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
    
    # Setup environment
    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0", rooms_in_row=3, rooms_in_col=4)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env) 
    env = DictResizeObs(env, (64, 64))
    
    seed = 218
    env.seed(seed)
    obs = env.reset()
    obs, _, done, _ = env.step(env.actions.forward)
    env.render(tile_size=64)
    
    # Initialize belief
    belief_zd = None
    prev_act_1h = None
    
    # Update belief from first observation
    belief_zd = wm_update_belief(wm, belief_zd, 
                               obs["image"].transpose(2,0,1) / 255.0, 
                               prev_act_1h)
    
    print("\n=== Testing Current Reconstruction ===")
    test_current_reconstruction(wm, belief_zd, obs)
    
    print("\n=== Testing Imagination from Current State ===")
    save_decoder_fixed(wm, belief_zd, obs, "initial")
    
    # Take a few steps and test again
    for step in range(3):
        prev_act_1h = onehot("forward", wm)
        obs, _, done, _ = env.step(env.actions.forward)
        belief_zd = wm_update_belief(wm, belief_zd, 
                                   obs["image"].transpose(2,0,1) / 255.0, 
                                   prev_act_1h)
        env.render(tile_size=64)
        time.sleep(0.5)
    
    print(f"\n=== After {3} Steps ===")
    test_current_reconstruction(wm, belief_zd, obs)
    save_decoder_fixed(wm, belief_zd, obs, "after_steps")
    
    # Run full step-by-step test
    print("\n=== Full Step-by-Step Test ===")
    step_by_step_reconstruction_test(wm, env, num_steps=5)
    
    env.close()