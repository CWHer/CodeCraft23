import argparse
import os
import random

from item_centric.agent import ItemBasedAgent
from item_centric.scheduler import GreedyScheduler
from RobotEnv.env_wrapper import EnvWrapper
from utils import fixSeed


def fitnessFunc(param):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    parser.add_argument("--no-statistics", default=False, action="store_true")
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
    item_delta = param[:7]
    station_type_delta = param[7:16]
    station_input_delta = param[16:]
    scheduler = GreedyScheduler({
        "item_delta": item_delta,
        "station_type_delta": station_type_delta,
        "station_input_delta": station_input_delta,
    })
    agent = ItemBasedAgent(scheduler)
    env_map = env.reset()
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)

    env.close()

    return -agent.moneys[-1]


if __name__ == "__main__":
    from cma import CMAEvolutionStrategy
    param = [100, 100, 100, 300, 300, 300, 900] \
        + [0, 0, 0, 0, 0, 0, 0, -50, -50] \
        + [0, 0, 0, 0, 0, 0, 0, 0]
    init_value = fitnessFunc(param)
    print("[CMA-ES] Initial value: ", init_value)
    es = CMAEvolutionStrategy(param, 10)
    es.optimize(
        fitnessFunc, iterations=100, verb_disp=10,
        callback=lambda x: print("[{}]".format(','.join(map(str, x.result[0]))))
    )
    value = fitnessFunc(es.result[0])
    print("[CMA-ES] Final result: ", es.result[0], value)
