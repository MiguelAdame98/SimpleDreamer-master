import gym
import numpy as np
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Make sure we have a ResizeObservation wrapper even on old Gym versions
# --------------------------------------------------------------------------
try:
    from gym.wrappers import ResizeObservation        # Gym >= 0.21
except (ImportError, AttributeError):
    # Gym < 0.21: roll our own ------------------------------------------------
    from gym import spaces
    import cv2

    class ResizeObservation(gym.ObservationWrapper):
        """
        Resize (H, W, C) observations to a user-defined (new_h, new_w).
        Works exactly like gym.wrappers.ResizeObservation that was added later.
        """
        def __init__(self, env, shape):
            super().__init__(env)
            if isinstance(shape, int):
                shape = (shape, shape)
            self.shape = tuple(shape)
            old_space = env.observation_space
            assert len(old_space.shape) == 3   # HWC
            self.observation_space = spaces.Box(
                low=old_space.low.min(),
                high=old_space.high.max(),
                shape=(self.shape[0], self.shape[1], old_space.shape[2]),
                dtype=old_space.dtype,
            )

        def observation(self, obs):
            # obs is H×W×C uint8 (after MiniGrid wrappers)
            obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
            return obs
# --------------------------------------------------------------------------
from dreamer.envs.wrappers import ChannelFirstEnv, PixelNormalization, SkipFrame



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

def make_minigrid_env(
        task_name="MiniGrid-ADRooms-Collision-v0",
        rooms_row=3,
        rooms_col=4,
        frame_skip=1,
        height=64,
        width=64,
        pixel_norm=True,
        run_dir= Path
        
    ):
        """
        Returns a Dreamer-ready MiniGrid environment that emits 56×56×3 images.

        Parameters
        ----------
        task_name   : str   Name registered in your fork (e.g. MiniGrid-*-v0)
        rooms_row   : int   passed to env constructor
        rooms_col   : int
        seed        : int   RNG seed
        frame_skip  : int   >1 will SkipFrame just like Atari helper
        pixel_norm  : bool  apply PixelNormalization (-0.5…0.5 floats)
        """
        import gym_minigrid
        from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        env = gym.make(
            task_name,
            rooms_in_row=rooms_row,
            rooms_in_col=rooms_col
        )
        # (7×7×3) → (56×56×3) is already done by your custom wrappers inside
        env = RGBImgPartialObsWrapper(env)   # adds "image" key
        env = ImgObsWrapper(env)             # drop everything except image

        env = gym.wrappers.ResizeObservation(env, (height, width))      # 64×64×3
        env = ChannelFirstEnv(env)                                     # 3×64×64
        env = SkipFrame(env, frame_skip)
        env = EpisodicStats(env)
        if os.getenv("DREAMER_RENDER") == "1":
            out_dir = (run_dir or Path.cwd() / "videos") / "videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            env = VideoEveryN(env, out_dir=out_dir, every=200)    
            env._video_prefix = task_name.replace("MiniGrid-", "")
            # --- generic Dreamer wrappers --------------------------------------
        if pixel_norm:
            env = PixelNormalization(env)            # float32 -0.5…0.5

        env.seed(1337)
        return env

class EpisodicStats(gym.Wrapper):
    """
    Adds `info["episodic_return"]` and `info["episodic_length"]`
    on the final `step()` of every episode.
    """
    def __init__(self, env):
        super().__init__(env)
        self._R = 0.0
        self._L = 0

    def reset(self, **kw):
        self._R, self._L = 0.0, 0
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._R += rew
        self._L += 1
        if done:
            info["episodic_return"]  = self._R
            info["episodic_length"]  = self._L
        return obs, rew, done, info
import imageio, numpy as np, datetime as dt, pathlib, gym

class VideoEveryN(gym.Wrapper):
    """
    Records an MP4 of every *N*-th episode by calling env.render("rgb_array").
    Works with **any** Gym version.
    """
    def __init__(self, env, out_dir, every=200, prefix="episode"):
        super().__init__(env)
        self.every      = every
        self.ep_counter = 0
        self.out_dir    = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, **kw):
        obs = super().reset(**kw)
        self.frames = [self.render(mode="rgb_array")]
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.frames.append(self.render(mode="rgb_array"))
        if done:
            self._maybe_write_video()
        return obs, rew, done, info

    # ------------------------------------------------------------------ #
    def _maybe_write_video(self):
        if self.ep_counter % self.every == 0:
            prefix = getattr(self, "_video_prefix", "episode")          # ← NEW line
            fname  = f"{prefix}_{self.ep_counter:05d}.mp4"
            path  = str(self.out_dir / fname)
            imageio.mimsave(path, self.frames, fps=10, macro_block_size=None)
            print(f"[video] saved {path}  ({len(self.frames)} f)")
        self.ep_counter += 1