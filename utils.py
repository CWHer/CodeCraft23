

def fixSeed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def parseLog(log_filename: str, output_filename) -> None:
    import pickle
    from task_utils import MetaTask

    with open(log_filename, "rb") as f:
        task_log = pickle.load(f)

    num_robots = 4
    assigned_tasks = [[] for _ in range(num_robots)]
    for log in task_log:
        task: MetaTask = log["task_info"]
        task.owned_item = False
        assigned_tasks[task.robot_id].append(task)
    with open(output_filename, "w") as f:
        f.write("from task_utils import MetaTask, TimeRange\n\n")
        f.write("assigned_tasks = [\n")
        for robot_id in range(4):
            f.write(
                f"    [{', '.join([str(task) for task in assigned_tasks[robot_id]])}],\n")
        f.write("]\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="map1.pkl")
    parser.add_argument("--output", type=str, default="map1_tasks.py")
    args = parser.parse_args()
    parseLog(args.log, args.output)
