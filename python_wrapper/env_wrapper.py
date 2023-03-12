import argparse
import multiprocessing
import os
import random
from typing import Dict, List, Optional, Tuple


class EnvWrapper:
    MAX_READ_SIZE = 4096
    END_TOKEN = "OK"

    def __init__(self, args) -> None:
        self.in_pipe_name = args.pipe_name + "_out"
        self.out_pipe_name = args.pipe_name + "_in"
        os.mkfifo(self.in_pipe_name)
        os.mkfifo(self.out_pipe_name)
        self.in_pipe = os.open(self.in_pipe_name,
                               os.O_SYNC | os.O_CREAT | os.O_RDWR)
        self.out_pipe = os.open(self.out_pipe_name,
                                os.O_SYNC | os.O_CREAT | os.O_RDWR)

        def launchEnv():
            command = f"{args.env_binary_name} {args.env_args} "\
                f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
            print(f"[INFO]: Launch Environment: {command}")
            os.system(f"{args.env_binary_name} {args.env_args} "
                      f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\"")

        self.env_process = multiprocessing.Process(target=launchEnv)
        self.env_process.start()

    @staticmethod
    def _parseObs(obs) -> Dict:
        # TODO: parse observation ...
        return obs

    def reset(self):
        # TODO: add parse observation ...
        observation = os.read(self.in_pipe, self.MAX_READ_SIZE * 10)
        return self._parseObs(observation)

    def recv(self) -> Tuple[Optional[Dict], float, bool, Dict]:
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return None
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # TODO: add reward ...
        # TODO: add parse observation ...
        observation = os.read(self.in_pipe, self.MAX_READ_SIZE)
        observation = observation.decode(encoding="ASCII")

        done = observation == self.END_TOKEN

        return self._parseObs(observation), 0.0, done, {}

    def send(self, actions: List[str]) -> None:
        os.write(self.out_pipe, '\n'.join(actions).encode(encoding="ASCII"))

    def close(self):
        self.env_process.join()

        os.close(self.in_pipe)
        os.close(self.out_pipe)
        os.remove(self.in_pipe_name)
        os.remove(self.out_pipe_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot")
    # TODO: FIXME: -d takes 20ms to execute, why?
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    args = parser.parse_args()
    print(args)

    from demo.simple_agent import SimpleAgent
    env = EnvWrapper(args)
    obs = env.reset()
    agent = SimpleAgent(obs)
    while True:
        obs, reward, done, info = env.recv()
        if done:
            break
        action = agent.act(obs)
        env.send(action)

    print("[INFO]: Waiting for environment to close...")
    env.close()

    print("[INFO]: Done!")
