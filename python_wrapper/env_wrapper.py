import argparse
import multiprocessing
import os
import random
from typing import Any, Dict, List, Optional, Tuple


class EnvWrapper:
    MAX_READ_SIZE = 4096
    END_TOKEN = "OK"

    def __init__(self, pipe_name, launch_command) -> None:
        self.in_pipe_name = pipe_name + "_out"
        self.out_pipe_name = pipe_name + "_in"

        self.launch_command = launch_command
        self.env_process = None
        self.in_pipe = None
        self.out_pipe = None

    def _cleanEnv(self):
        if self.env_process is not None:
            self.env_process.terminate()

    def _makePipe(self):
        os.mkfifo(self.in_pipe_name)
        os.mkfifo(self.out_pipe_name)
        pipe_flag = os.O_SYNC | os.O_CREAT | os.O_RDWR
        self.in_pipe = os.open(self.in_pipe_name, pipe_flag)
        self.out_pipe = os.open(self.out_pipe_name, pipe_flag)

    def _removePipe(self):
        if self.in_pipe is not None \
                and self.out_pipe is not None:
            os.close(self.in_pipe)
            os.close(self.out_pipe)
            os.remove(self.in_pipe_name)
            os.remove(self.out_pipe_name)

    @staticmethod
    def _parseObs(obs) -> Optional[Dict]:
        obs = obs.decode(encoding="ASCII")
        if obs == EnvWrapper.END_TOKEN:
            return None

        def _parseBaseInfo(lines: List[str]):
            frame_id, money = tuple(map(int, lines.pop(0).split()))
            return {
                "frame_id": frame_id,
                "money": money,
            }

        def _parseStation(lines: List[str]):
            station_specs = {
                "station_type": int,
                "loc_x": float,
                "loc_y": float,
                "remain_time": int,
                "input_status": int,
                "output_status": int,
            }

            num_stations = int(lines.pop(0))
            stations = []
            for _ in range(num_stations):
                items = lines.pop(0).split()
                station = {
                    key: station_specs[key](value)
                    for key, value in zip(station_specs.keys(), items)
                }
                stations.append(station)
            return {"stations": stations}

        def _parseRobots(lines: List[str]):
            robot_specs = {
                "station_id": int,
                "item_type": int,
                "time_coef": float,
                "momentum_coef": float,
                "angular_speed": float,
                "line_speed_x": float,
                "line_speed_y": float,
                "theta": float,
                "loc_x": float,
                "loc_y": float,
            }
            robots = []
            for _ in range(4):
                items = lines.pop(0).split()
                robot = {
                    key: robot_specs[key](value)
                    for key, value in zip(robot_specs.keys(), items)
                }
                robots.append(robot)
            return {"robots": robots}

        obs_dict: Dict[str, Any] = {}
        lines = obs.split('\n')
        obs_dict.update(_parseBaseInfo(lines))
        obs_dict.update(_parseStation(lines))
        obs_dict.update(_parseRobots(lines))
        return obs_dict

    def reset(self):
        self._cleanEnv()
        self._removePipe()
        self._makePipe()

        self.env_process = multiprocessing.Process(
            target=lambda cmd: os.system(cmd), args=(self.launch_command, ))
        print(f"[INFO]: Launch Environment: {self.launch_command}")
        self.env_process.start()

        assert self.in_pipe is not None
        observation = os.read(self.in_pipe, self.MAX_READ_SIZE * 10)
        return {"map": observation.decode(encoding="ASCII")}

    def recv(self) -> Tuple[Optional[Dict], bool]:
        """
        Returns:
            observation (object): agent's observation of the current environment
            done (boolean): whether the episode has ended, in which case further step() calls will return None
        """
        assert self.in_pipe is not None
        observation = os.read(self.in_pipe, self.MAX_READ_SIZE)
        obs = self._parseObs(observation)
        return obs, obs is None

    def send(self, actions: List[str]) -> None:
        assert self.out_pipe is not None
        os.write(self.out_pipe, '\n'.join(actions).encode(encoding="ASCII"))

    def close(self):
        self._cleanEnv()
        self._removePipe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    args = parser.parse_args()
    print(args)

    from demo.simple_agent import SimpleAgent
    launch_command = f"{args.env_binary_name} {args.env_args} "\
        f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
    env = EnvWrapper(args.pipe_name, launch_command)
    obs = env.reset()
    agent = SimpleAgent(obs)
    for _ in range(100):
        obs, done = env.recv()
        assert not done
        action = agent.act(obs)
        env.send(action)

    for _ in range(2):
        obs = env.reset()
        while True:
            obs, done = env.recv()
            if done:
                break
            action = agent.act(obs)
            env.send(action)

    print("[INFO]: Waiting for environment to close...")
    env.close()

    print("[INFO]: Done!")
