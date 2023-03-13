from typing import List


class SimpleAgent:
    def __init__(self, observation, info_log=False):
        self.info_log = info_log
        print("[INFO]: SimpleAgent init")
        print("[Py] Observation: ", observation, sep='\n')

    def act(self, observation) -> List[str]:
        if self.info_log:
            print("[INFO]: SimpleAgent act")
            print("[Py] Observation: ", observation, sep='\n')
        line_speed, angle_speed = 3, 1.5
        action = []
        for robot_id in range(4):
            action.append(f"forward {robot_id} {line_speed}")
            action.append(f"rotate {robot_id} {angle_speed}")
        if self.info_log:
            print("[Py] Action: ", action, sep='\n')
        return action


if __name__ == "__main__":
    agent = SimpleAgent("none")
    for _ in range(5):
        agent.act("none")
