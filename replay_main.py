import argparse
import os
import random

from item_centric.agent import ReplayAgent
from RobotEnv.env_wrapper import EnvWrapper
from utils import fixSeed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    robot_env_path = os.path.join(os.path.dirname(__file__), "RobotEnv")
    args.map_id = os.path.join(robot_env_path, args.map_id)
    args.env_binary_name = os.path.join(robot_env_path, args.env_binary_name)
    args.env_wrapper_name = os.path.join(robot_env_path, args.env_wrapper_name)
    print(args)

    launch_command = f"{args.env_binary_name} {args.env_args} "\
        f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
    env = EnvWrapper(args.pipe_name, launch_command)

    fixSeed(args.seed)
    env_map = env.reset()["map"]
    map1_line = ".....................................4..4..A..7..7..7..A..6..6......................................"
    map2_line = ".5......................8....1.................A.4.A.................1....8.......................5."
    map3_line = ".4......4.....4...................................................................6.....6.....6....."
    map4_line = "..............2..................................2..................................2..............."
    if map1_line in env_map:
        import map1_tasks
        assigned_tasks = map1_tasks.assigned_tasks
    elif map2_line in env_map:
        import map2_tasks
        assigned_tasks = map2_tasks.assigned_tasks
    elif map3_line in env_map:
        import map3_tasks
        assigned_tasks = map3_tasks.assigned_tasks
    elif map4_line in env_map:
        import map4_tasks
        assigned_tasks = map4_tasks.assigned_tasks
    else:
        raise ValueError("Unknown map")
    agent = ReplayAgent(assigned_tasks)
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)

    env.close()
