# dreamer_mg/world_model_utils.py
# ----------------------------------------------------------------------
#  ✧  World-model utilities for MiniGrid collision prediction & planning
# ----------------------------------------------------------------------
import yaml, torch, pathlib, heapq, random, numpy as np
from collections import namedtuple
from dreamer.modules.encoder import Encoder
from dreamer.modules.model   import RSSM

# ─────────────────────────────── constants ────────────────────────────
State    = namedtuple("State", ["x", "y", "d"])          # planner state
DIR_VECS = [(1,0), (0,1), (-1,0), (0,-1)]                # 0:right 1:down …

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════
# 1.  LOADING  (one-liner:  wm = load_world_model("…/iter0500.pt") )
# ══════════════════════════════════════════════════════════════════════
def load_world_model(ckpt_path: str,
                     config_yaml: str = "configs/minigrid-default.yml"):
    """
    Returns an object with .encoder and .rssm already on the correct device.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg  = yaml.safe_load(open(config_yaml))

    enc  = Encoder((3, 64, 64), cfg).to(DEVICE)
    rssm = RSSM(action_size=3, config=cfg).to(DEVICE)

    enc .load_state_dict(ckpt["modules"]["encoder"]);  enc .eval()
    rssm.load_state_dict(ckpt["modules"]["rssm"   ]);  rssm.eval()

    wm = lambda: None             # poor-man struct
    wm.encoder = enc
    wm.rssm    = rssm
    return wm

# ══════════════════════════════════════════════════════════════════════
# 2.  COLLISION PREDICTION  (wm_predict_collision)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def wm_predict_collision(wm,
                         frame_rgb: np.ndarray,
                         action_seq: list[str],
                         threshold: float = 0.5,
                         num_rollouts: int = 5) -> bool:
    """
    True  → treat prefix as unsafe
    False → prefix considered safe  (no collision predicted)
    """
    # current latent (z₀, d₀) ------------------------------------------
    img = torch.tensor(frame_rgb, dtype=torch.float32,
                       device=DEVICE).unsqueeze(0)         # (1,3,64,64)
    emb      = wm.encoder(img).view(1, -1)
    z, d_lat = wm.rssm.representation_model(
                  emb, wm.rssm.recurrent_model_input_init(1)[1])

    # one-hot actions ---------------------------------------------------
    table = {'forward':[1,0,0], 'right':[0,1,0], 'left':[0,0,1]}
    acts  = torch.tensor([table[a] for a in action_seq],
                         dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # Monte-Carlo roll-outs --------------------------------------------
    hits = 0
    for _ in range(num_rollouts):
        z_s, d_s = z, d_lat
        for t, a in enumerate(action_seq):
            d_s = wm.rssm.recurrent_model(z_s, acts[:, t], d_s)
            _, z_s = wm.rssm.transition_model(d_s)

            if a == "forward":
                # TODO ► replace by your future collision-head
                pseudo_prob = 0.15
                if random.random() < pseudo_prob:
                    hits += 1;  break

    return (hits / num_rollouts) >= threshold

# ══════════════════════════════════════════════════════════════════════
# 3.  A*  PLANNER  (astar_prims)  –  uses wm_predict_collision
# ══════════════════════════════════════════════════════════════════════
def heuristic(s: State, g: State) -> int:
    manh = abs(s.x - g.x) + abs(s.y - g.y)
    turn = min((s.d - g.d) % 4, (g.d - s.d) % 4)
    return manh + turn

def astar_prims(wm,
                frame_rgb: np.ndarray,
                start: State,
                goal:  State,
                num_rollouts: int = 8,
                verbose: bool = False) -> list[str]:
    """
    Returns list of acts ["forward","left",…]  OR  [] if no safe path found.
    """
    pq      = [(heuristic(start, goal), 0, start, [])]      # (f,g,state,seq)
    g_score = {start: 0}
    closed  = set()

    while pq:
        f, g, (x,y,d), seq = heapq.heappop(pq)
        if verbose:
            print(f"[A*] pop  {State(x,y,d)}  g={g} f={f}  seq={seq}")

        if (x,y,d) in closed:          # stale entry
            continue
        closed.add((x,y,d))

        if (x,y,d) == (goal.x, goal.y, goal.d):
            return seq

        for act in ("left","right","forward"):
            if act == "forward":
                dx,dy = DIR_VECS[d];  nx,ny,nd = x+dx, y+dy, d
            elif act == "left":
                nx,ny,nd = x, y, (d-1) % 4
            else:
                nx,ny,nd = x, y, (d+1) % 4

            ns = State(nx,ny,nd)
            if ns in closed:
                continue

            new_seq = seq + [act]

            if wm_predict_collision(wm, frame_rgb, new_seq,
                                    num_rollouts=num_rollouts):
                if verbose:  print("   skip (collision)")
                continue

            g2, h2 = g+1, heuristic(ns, goal)
            f2     = g2 + h2
            if g2 < g_score.get(ns, float("inf")):
                g_score[ns] = g2
                heapq.heappush(pq, (f2, g2, ns, new_seq))

    return []                      # no safe path found

# ══════════════════════════════════════════════════════════════════════
# 4.  QUICK SELF-TEST  (python -m dreamer_mg.world_model_utils)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import cv2, os
    ckpt = "runs/mg_collision/ckpt/iter0500.pt"
    if not os.path.exists(ckpt):
        print("⚠  checkpoint not found – demo skipped")
        exit()

    wm = load_world_model(ckpt)
    dummy = np.zeros((3,64,64), dtype=np.float32)           # fake frame

    start = State(0,0,0);  goal = State(3,3,0)
    path  = astar_prims(wm, dummy, start, goal,
                        num_rollouts=3, verbose=True)
    print("→ planned path:", path or "NONE (blocked)")
