import os, argparse
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path          #  ←  add this import
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory, new_run_dir

# ⬇️  NEW: include make_minigrid_env
from dreamer.envs.envs import (
    make_atari_env, make_minigrid_env, get_env_infos
)


def main(config_file):
    config = load_config(config_file)
    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    run_dir = new_run_dir(exp_name="mg_collision")
    shutil.copy(config_file, run_dir / "config.yml")   # save exact cfg
    writer   = SummaryWriter(run_dir)
    device = config.operation.device

    if config.environment.benchmark == "minigrid":          # ⬅️ NEW
        env = make_minigrid_env(
            task_name   = config.environment.task_name,
            rooms_row   = config.environment.rooms_row,
            rooms_col   = config.environment.rooms_col,
            frame_skip  = config.environment.frame_skip,
            pixel_norm  = config.environment.pixel_norm,
            run_dir=run_dir

        )
    else:
        raise ValueError(f"Unknown benchmark: {config.environment.benchmark}")
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)
    print(env.action_space)

    
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config,run_dir
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    agent.train(env)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minigrid-default.yml",
        help="Path to the YAML config file",
    )
    main(parser.parse_args().config)
