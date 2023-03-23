import argparse

from item_centric.agent import ItemBasedAgent
from item_centric.scheduler import GreedyScheduler
from judge_env import JudgeEnv
from utils import fixSeed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-statistics",
                        default=False, action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    env = JudgeEnv()
    fixSeed(args.seed)
    scheduler = GreedyScheduler()
    agent = ItemBasedAgent(scheduler)
    env_map = env.reset()
    env._writeDone()
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)

    if args.show_statistics:
        agent.showStatistics()
