from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from utils import concatDict


class FakeVenv:
    def __init__(self, env_fns: List[Callable]) -> None:
        self.envs = [env_fn() for env_fn in env_fns]

    def reset(self, env_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        if env_ids is None:
            env_ids = np.arange(len(self.envs))
        obs, info = zip(*[self.envs[i].reset() for i in env_ids])
        obs = np.stack(obs, axis=0)
        info = concatDict(info)
        return obs, info

    def step(self, actions: List[Any], env_ids: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        obs, rews, dones, info = zip(
            *[self.envs[i].step(action)
              for i, action in zip(env_ids, actions)]
        )
        obs = np.stack(obs, axis=0)
        rews = np.stack(rews, axis=0)
        dones = np.stack(dones, axis=0)
        info = concatDict(info)
        return obs, rews, dones, info

    def close(self):
        for env in self.envs:
            env.close()

    def __len__(self) -> int:
        return len(self.envs)


if __name__ == "__main__":
    import argparse
    import os
    import random

    # test wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    args = parser.parse_args()
    robot_env_path = os.path.join(os.path.dirname(__file__), "RobotEnv")
    args.map_id = os.path.join(robot_env_path, args.map_id)
    args.env_binary_name = os.path.join(robot_env_path, args.env_binary_name)
    args.env_wrapper_name = os.path.join(robot_env_path, args.env_wrapper_name)
    print(args)

    from env_wrapper import EnvWrapper, ObsWrapper, RewardWrapper

    def makeRobotEnv(pipe_name: str):
        launch_command = f"{args.env_binary_name} {args.env_args} "\
            f"-m {args.map_id} \"{args.env_wrapper_name} {pipe_name}\""
        env = EnvWrapper(pipe_name, launch_command)
        env = ObsWrapper(env)
        env = RewardWrapper(env)
        return env

    import functools

    num_envs = 10
    vec_env = FakeVenv(
        [functools.partial(makeRobotEnv, f"{args.pipe_name}_{i}")
         for i in range(num_envs)]
    )

    obs, info = vec_env.reset()
    env_ids = np.arange(num_envs)
    for _ in range(100):
        action = [["forward 0 1"] for _ in range(num_envs)]
        obs, reward, done, info = vec_env.step(action, env_ids)
    vec_env.close()
