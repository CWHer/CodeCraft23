from typing import List


class SimpleAgent:
    def __init__(self, observation):
        print("[INFO]: SimpleAgent init")
        print("Observation: ", observation, sep='\n')

    def act(self, observation) -> List[str]:
        print("[INFO]: SimpleAgent act")
        print("Observation: ", observation, sep='\n')
        line_speed, angle_speed = 3, 1.5
        action = []
        for robot_id in range(4):
            action.append(f"forward {robot_id} {line_speed}")
            action.append(f"rotate {robot_id} {angle_speed}")
        print("Action: ", action, sep='\n')
        return action


if __name__ == "__main__":
    agent = SimpleAgent("none")
    for _ in range(5):
        agent.act("none")
