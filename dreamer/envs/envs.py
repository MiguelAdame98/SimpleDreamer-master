import gym
import dmc2gym

from dreamer.envs.wrappers import *
import gym
import gym_minigrid 
from gym_minigrid.wrappers import ImgActionObsWrapper, RGBImgPartialObsWrapper
from gym_minigrid.window import Window




def make_atari_env(task_name, skip_frame, width, height, seed, pixel_norm=True):
    env = gym.make(task_name)
    env = gym.wrappers.ResizeObservation(env, (height, width))
    env = ChannelFirstEnv(env)
    env = SkipFrame(env, skip_frame)
    if pixel_norm:
        env = PixelNormalization(env)
    env.seed(seed)
    return env


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size

def make_minigrid_env(env):
    #--- ENV INIT ---#
        env_name = 'MiniGrid-4-tiles-ad-rooms-v0'
        print(env_name)
        env = gym.make(env_name, 3, 4)
        import inspect, sys
        print("ENV CODE =", inspect.getfile(env.__class__))
        env = RGBImgPartialObsWrapper(env)
        env = ImgActionObsWrapper(env)
        window = Window(env_name)
        seed = seed

