import time
import gym_minigrid, gym
import random
from gym_minigrid.minigrid import Wall

env = gym.make(
    "MiniGrid-ADRooms-Collision-v0",
    rooms_in_row=3,
    rooms_in_col=4
)

obs = env.reset()
done = False

FPS = 6                    # frames per second you want to SEE
dt  = 3.0 / FPS            # seconds per frame

env.render(tile_size=64)   # show the initial state once
SAFE_STEPS=5
steps=0
while not done:
    steps+=1
    front_pos = env.front_pos           # (x, y) tuple
    front_cell = env.grid.get(*front_pos)
    if steps < SAFE_STEPS:
            if isinstance(front_cell, Wall):
                        # there's a wall ahead → turn
                env_act = random.choice([0, 1])  # 0=left, 1=right
            else:
                env_act = 2                      # 2=forward
    else:
        # after SAFE_STEPS, pure random
        env_act = random.randrange(0,2)           # TODO: your policy
    
    obs, reward, done, info = env.step(env_act)    # legacy 4-tuple API
    env.render(tile_size=64)                      # draw AFTER the step
    time.sleep(dt)                                # slow things down

env.close()