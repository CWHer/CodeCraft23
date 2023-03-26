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
    parser.add_argument("--use_revised_control", default=True, type=bool)
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
        "item_delta": np.array([185.83115802612176, 157.22915455033558, 133.60740480498708, 419.09446674621876, 410.2535861029388, 252.01785224150618, 949.8240613739746]),
        "station_type_delta": np.array([-5.330061794655554, -0.7045912398454199, 18.273680647867245, 14.979175210163486, -8.621498668856837, 22.95505120597874, 9.581022704586264, -21.39271866842179, -32.60624101790579]),
        "station_input_delta": np.array([-18.45155876666678, 12.624273091167616, 48.943908391100486, -10.99797089825291, -17.154114166184037, 33.08042211800793, 23.79039782556589, 8.375975035892013]),
    }
    params3: Dict[str, np.ndarray] = {
        "item_delta": np.array([133.92005899791587, 111.18427167185239, 119.41513851316218, 322.26850599354674, 286.357413279066, 258.28838391514296, 915.059225678683]),
        "station_type_delta": np.array([-8.652382028640252, 1.5112569732492247, -18.87461563435858, 8.946392974677407, -6.359663597206546, 39.93042285474438, 22.016960188261702, -58.424824114590834, -49.761557791806474]),
        "station_input_delta": np.array([-2.2296154308920144, -18.252745342786497, -18.00385139293702, 11.771218646571702, -43.41683890123609, 17.060287254906946, 22.106733561523914, -2.158863514818551]),
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
        args.use_revised_control = False
    elif map2_line in env_map:
        params = params2
        args.use_revised_control = True
    elif map3_line in env_map:
        params = params3
        args.use_revised_control = True
    elif map4_line in env_map:
        params = params4
        args.use_revised_control = False
    else:
        raise ValueError("Unknown map")
    scheduler = GreedyScheduler(params)
    agent = ItemBasedAgent(scheduler, args.use_revised_control)
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)

    env.close()

    if not args.no_statistics:
        agent.showStatistics(args.save_unfinished)
