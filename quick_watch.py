# quick_watch.py
import time
import random
import gym
import gym_minigrid
from gym_minigrid.minigrid import Wall
from gym_minigrid.register import register as mg_register, env_list

# Import the module that defines AisleDoorRooms so its class + methods are loaded
import gym_minigrid.envs.aisle_door_rooms  

# Ensure IDs exist but don't double-register
ENV_ID   = "MiniGrid-4-tiles-ad-rooms-v0"
ALIAS_ID = "MiniGrid-ADRooms-Collision-v0"
if ENV_ID not in env_list:
    mg_register(id=ENV_ID, entry_point="gym_minigrid.envs.aisle_door_rooms:AisleDoorRooms")
if ALIAS_ID not in env_list:
    mg_register(id=ALIAS_ID, entry_point="gym_minigrid.envs.aisle_door_rooms:AisleDoorRooms")

# Build the env (prefer the 4-tiles id you asked for)
try:
    env = gym.make(ENV_ID, rooms_in_row=3, rooms_in_col=4, max_steps=None)
except Exception:
    env = gym.make(ALIAS_ID, rooms_in_row=3, rooms_in_col=4, max_steps=None)

print("Wrapper class:", env.__class__.__module__, env.__class__.__name__)
print("Base class   :", env.unwrapped.__class__.__module__, env.unwrapped.__class__.__name__)
print("gym_minigrid :", getattr(gym_minigrid, "__file__", "?"))
print("Has MutateConnectivity → wrapper:", hasattr(env, "MutateConnectivity"),
      " base:", hasattr(env.unwrapped, "MutateConnectivity"))

# Always call custom methods on the base (wrappers may hide attrs)
base = env.unwrapped
if not hasattr(base, "MutateConnectivity"):
    raise RuntimeError("MutateConnectivity not found on base env. "
                       "Confirm it's defined inside AisleDoorRooms in aisle_door_rooms.py")

# Optional: deterministic demo
random.seed(42)
try:
    env.seed(42)
except Exception:
    pass

# Reset & initial render
out = env.reset()
if isinstance(out, tuple):
    obs = out[0]
else:
    obs = out
done = False
env.render(tile_size=64)

# Simple stepper (gym vs gymnasium compatible)
def step_once(e, step_idx, safe_steps=5):
    front_pos = e.front_pos
    front_cell = e.grid.get(*front_pos)
    if step_idx < safe_steps:
        act = random.choice([0, 1]) if isinstance(front_cell, Wall) else 2
    else:
        act = random.randrange(0, 3)
    out = e.step(act)
    if len(out) == 4:
        obs, reward, done, info = out
    else:
        obs, reward, term, trunc, info = out
        done = bool(term or trunc)
    return obs, reward, done, info

def run_phase(e, n_steps=80, fps=6):
    dt = 3.0 / fps
    done = False
    steps = 0
    while not done and steps < n_steps:
        steps += 1
        _, _, done, _ = step_once(e, steps, safe_steps=5)
        e.render(tile_size=64)
        time.sleep(dt)

# PHASE 0: baseline
print("\n[PHASE 0] Baseline walk")
run_phase(env, n_steps=40, fps=6)

# PHASE 1: front obstacles (non-blocking)
print("\n[PHASE 1] Front obstacles (non-blocking, 1 tile inside rooms)")
base.MutateConnectivity(cutoff_rooms_rate=0.0, front_obstacle_rate=0.55, seed=123)
run_phase(env, n_steps=40, fps=6)

# PHASE 2: hard cuts (block both ends of incident corridors for some rooms)
print("\n[PHASE 2] Hard cuts (isolate ~30% of rooms)")
base.MutateConnectivity(cutoff_rooms_rate=0.30, front_obstacle_rate=0.0, seed=999)
run_phase(env, n_steps=40, fps=6)

# PHASE 3: remove and walk again (requires your precise Remove_Obstacles)
if hasattr(base, "Remove_Obstacles"):
    print("\n[PHASE 3] Remove obstacles → restore original cells")
    base.Remove_Obstacles()
    run_phase(env, n_steps=40, fps=6)
else:
    print("[WARN] Remove_Obstacles not found on base env; skipping restore")

env.close()
print("\n[Done]")
