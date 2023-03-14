from typing import Dict, List, Tuple

import numpy as np
from RobotEnv.env_wrapper import EnvWrapper


class ObsWrapper():
    def __init__(self, env: EnvWrapper):
        self.env = env

    @staticmethod
    def _toNumpy(obs: Dict) -> np.ndarray:
        assert obs is not None
        RESERVED_KEYS = ["frame_id", "money", "stations", "robots"]
        assert set(obs.keys()) == set(RESERVED_KEYS)
        arr = [[obs["frame_id"], obs["money"]]]
        for station in obs["stations"]:
            arr.append(list(station.values()))
        for robot in obs["robots"]:
            arr.append(list(robot.values()))
        arr = np.concatenate(arr, dtype=np.float32)
        return arr

    def reset(self) -> Tuple[np.ndarray, Dict]:
        game_map = self.env.reset()
        obs, done = self.env.recv()
        assert not done
        info = {
            "frame_id": obs["frame_id"],
            "money": obs["money"],
            "game_map": game_map,
        }
        obs = self._toNumpy(obs)
        return obs, info

    def step(self, action: List[str]) -> Tuple[np.ndarray, bool, Dict]:
        self.env.send(action)
        obs, done = self.env.recv()
        info = {
            "frame_id": obs["frame_id"],
            "money": obs["money"],
        }
        obs = self._toNumpy(obs) if not done else np.array([])
        return obs, done, info

    def close(self):
        self.env.close()


class RewardWrapper():
    def __init__(self, env):
        self.env = env
        self.last_money = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        done, info = self.env.reset()
        self.last_money = info["money"]
        return done, info

    def step(self, action: List[str]) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        obs, done, info = self.env.step(action)
        reward = info["money"] - self.last_money
        self.last_money = info["money"]
        return obs, reward, done, info

    def close(self):
        self.env.close()


if __name__ == "__main__":
    import os
    import random
    import argparse

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

    launch_command = f"{args.env_binary_name} {args.env_args} "\
        f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
    env = EnvWrapper(args.pipe_name, launch_command)
    env = ObsWrapper(env)
    env = RewardWrapper(env)

    obs, info = env.reset()
    for _ in range(100):
        action = ["forward 0 1"]
        obs, reward, done, info = env.step(action)
    env.close()
