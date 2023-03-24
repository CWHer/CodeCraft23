import sys
from typing import Any, Dict, List, Optional, Tuple


class JudgeEnv:
    END_TOKEN = "OK"

    def __init__(self) -> None:
        self.frame_id = 0
        self.env_map = []

    def reset(self):
        while True:
            line = sys.stdin.readline().strip()
            if line == self.END_TOKEN:
                break
            self.env_map.append(line)
        return self.env_map
        # print(self.env_map, file=sys.stderr)

    def _writeDone(self):
        sys.stdout.write(f"{self.END_TOKEN}\n")
        sys.stdout.flush()

    @staticmethod
    def _parseObs(obs) -> Optional[Dict]:
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
        obs_dict.update(_parseBaseInfo(obs))
        obs_dict.update(_parseStation(obs))
        obs_dict.update(_parseRobots(obs))
        return obs_dict

    def recv(self) -> Tuple[Optional[Dict], bool]:
        """
        Returns:
            observation (object): agent's observation of the current environment
            done (boolean): whether the episode has ended, in which case further step() calls will return None
        """
        observation = []
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                return None, True
            if line == self.END_TOKEN:
                break
            observation.append(line)
        # print(observation, file=sys.stderr)
        self.frame_id = int(observation[0].split()[0])
        obs = self._parseObs(observation)
        return obs, False

    def send(self, actions: List[str]) -> None:
        sys.stdout.write(f"{self.frame_id}\n")
        for action in actions:
            sys.stdout.write(f"{action}\n")
        self._writeDone()


if __name__ == "__main__":
    print("[INFO]: Launching env wrapper", file=sys.stderr)
    env = JudgeEnv()
    env_map = env.reset()
    # print(env_map, file=sys.stderr)
    env._writeDone()

    while True:
        obs, done = env.recv()
        # print(obs, file=sys.stderr)
        if done:
            break
        env.send(["forward 0 1"])

    print("[INFO]: Env wrapper finished", file=sys.stderr)
