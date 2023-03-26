import argparse

from judge_env import JudgeEnv
from replay_agent import ReplayAgent
from utils import fixSeed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    env = JudgeEnv()
    fixSeed(args.seed)
    env_map = env.reset()
    map1_line = ".....................................4..4..A..7..7..7..A..6..6......................................"
    map2_line = ".5......................8....1.................A.4.A.................1....8.......................5."
    map3_line = ".4......4.....4...................................................................6.....6.....6....."
    map4_line = "..............2..................................2..................................2..............."
    if map1_line in env_map:
        import map1_tasks
        assigned_tasks = map1_tasks.assigned_tasks
        from subtask_to_action_revised import SubtaskToAction
        subtask_to_action = SubtaskToAction()
    elif map2_line in env_map:
        import map2_tasks
        assigned_tasks = map2_tasks.assigned_tasks
        from subtask_to_action_revised import SubtaskToAction
        subtask_to_action = SubtaskToAction()
    elif map3_line in env_map:
        import map3_tasks
        assigned_tasks = map3_tasks.assigned_tasks
        from subtask_to_action_revised import SubtaskToAction
        subtask_to_action = SubtaskToAction()
    elif map4_line in env_map:
        import map4_tasks
        assigned_tasks = map4_tasks.assigned_tasks
        from subtask_to_action_revised import SubtaskToAction
        subtask_to_action = SubtaskToAction()
    else:
        raise ValueError("Unknown map")
    agent = ReplayAgent(assigned_tasks, subtask_to_action)
    env._writeDone()
    while True:
        obs, done = env.recv()
        if done:
            break
        actions = agent.step(obs)
        env.send(actions)
