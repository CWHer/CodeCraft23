import argparse
import os
import random
from typing import Dict

import numpy as np

from item_centric.agent import ItemBasedAgent
from item_centric.scheduler import BaseScheduler, GreedyScheduler
# from robot_centric.agent import RobotBasedAgent
# from robot_centric.scheduler import GreedyScheduler
from RobotEnv.env_wrapper import EnvWrapper
from utils import fixSeed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    parser.add_argument("--no-statistics", default=False, action="store_true")
    parser.add_argument("--save-unfinished", default=False, action="store_true")
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
    params1: Dict[str, np.ndarray] = {
        "item_delta": np.array([80.3376284386988, 84.96612499910397, 90.15230480952756, 295.97081096154506, 309.6957147410033, 316.14641529438376, 943.4366759383134]),
        "station_type_delta": np.array([-0.30460806173184807, 2.961267537773305, -18.61857700084019, -14.027662318030309, 11.413320208682865, -5.041639498221187, -1.0569872226983055, -58.04906263000672, -57.894013793638806]),
        "station_input_delta": np.array([-23.520327048948232, 36.35279187034496, 10.57833381897524, 7.453001043696904, 2.5394075148521797, 16.342214584460354, -25.5388309850334, -9.97643019133318]),
    }
    params2: Dict[str, np.ndarray] = {
        "item_delta": np.array([99.39320089237772, 90.21598770408865, 126.8026247765484, 295.2322009469579, 308.39283305174195, 329.4354281402235, 911.8557691007792]),
        "station_type_delta": np.array([-24.897367748729074, 17.76691660043598, -27.409195044991616, -1.9045650590291228, 25.33720591216186, -6.224080784453157, 3.4603294449650814, -40.312197676211824, -27.843026973189815]),
        "station_input_delta": np.array([5.027318962759688, -29.91346770249244, 30.001141339554206, -15.719764870219747, -6.421682561154453, 1.2298210401204237, 0.7666137102878348, 4.799258963723895]),
    }
    params3: Dict[str, np.ndarray] = {
        "item_delta": np.array([121.84749957896385, 53.89114584581402, 111.83930017900462, 331.55028919146565, 277.2676412634333, 266.9454843275944, 933.829877702463]),
        "station_type_delta": np.array([-27.227823596322857, 1.287945818504169, -0.39668986021366415, 30.34769073944965, 5.167529951653974, 38.428551278159745, -65.86227077181675, -39.870464756212954, -37.13625417307464]),
        "station_input_delta": np.array([-1.344184243868808, -15.500603108090463, -7.375048408151203, 31.310929178324006, -62.1826089723314, -24.459317162997102, -46.96950147809197, -8.694602010526792]),
    }
    params4: Dict[str, np.ndarray] = {
        "item_delta": np.array([60.01318832567292, 115.14177498951736, 97.18244513161952, 333.10347326567273, 311.8929172895734, 279.2606865189081, 909.6281147676317]),
        "station_type_delta": np.array([17.7308836624418, -16.189916508419536, -17.005874995492345, 17.87598742969229, 4.992385329845929, 16.35826451121697, -2.397166111847363, -3.781890288239909, 0.14420203515653007]),
        "station_input_delta": np.array([-0.5735702863521599, -23.64876933192046, 55.28736126304761, -28.745102044402994, 16.489855060759258, -7.159591969873366, -24.275789827767316, 86.09890123049644]),
    }
    map1_line = ".....................................4..4..A..7..7..7..A..6..6......................................"
    map2_line = ".5......................8....1.................A.4.A.................1....8.......................5."
    map3_line = ".4......4.....4...................................................................6.....6.....6....."
    map4_line = "..............2..................................2..................................2..............."
    if map1_line in env_map:
        params = params1
    elif map2_line in env_map:
        params = params2
    elif map3_line in env_map:
        params = params3
    elif map4_line in env_map:
        params = params4
    else:
        raise ValueError("Unknown map")
    scheduler = GreedyScheduler(params)
    agent = ItemBasedAgent(scheduler)
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)

    env.close()

    if not args.no_statistics:
        agent.showStatistics(args.save_unfinished)
