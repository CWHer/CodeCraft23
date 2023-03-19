import argparse
import os
import random

from RobotEnv.env_wrapper import EnvWrapper
from scheduler import RandomScheduler
from subtask_to_action import SubtaskToAction
from task_checker import TaskChecker
from task_to_subtask import TaskManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    parser.add_argument("--show-statistics", default=True)
    args = parser.parse_args()
    robot_env_path = os.path.join(os.path.dirname(__file__), "RobotEnv")
    args.map_id = os.path.join(robot_env_path, args.map_id)
    args.env_binary_name = os.path.join(robot_env_path, args.env_binary_name)
    args.env_wrapper_name = os.path.join(robot_env_path, args.env_wrapper_name)
    print(args)

    launch_command = f"{args.env_binary_name} {args.env_args} "\
        f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
    env = EnvWrapper(args.pipe_name, launch_command)

    num_robots = 4
    action_generator = SubtaskToAction()
    task_manager = TaskManager()
    task_checker = TaskChecker()
    scheduler = RandomScheduler()

    moneys, task_log = [], []
    start_time = [0 for _ in range(num_robots)]

    obs = env.reset()
    while True:
        obs, done = env.recv()
        if done:
            break
        assert obs is not None
        moneys.append(obs["money"])

        # check task status
        subtasks = [None] * num_robots
        for i, task in enumerate(scheduler.stat()):
            if task is None:
                continue
            task.update(obs)
            subtask = task_manager.makeSubtask(task, obs)
            if task_manager.isTaskDone(task, subtask):
                task_log.append({
                    "cycle": obs["frame_id"],
                    "duration": obs["frame_id"] - start_time[i],
                    "task_info": task
                })
                continue
            subtasks[i] = subtask

        # make decision
        robot_indices = [
            i for i in range(num_robots)
            if subtasks[i] is None
        ]  # idle robots
        scheduler.clear(robot_indices)
        candidate_tasks = task_checker.genTasks(obs)
        candidate_tasks = task_checker.filterInvalidTasks(candidate_tasks, obs)
        for i in range(num_robots):
            if subtasks[i] is None:
                robot_tasks = task_checker.checkConflict(
                    candidate_tasks[i], scheduler.stat(), obs)
                selected_task = scheduler.select(i, robot_tasks, obs)
                subtask = task_manager.makeSubtask(selected_task, obs)
                subtasks[i] = subtask
                start_time[i] = obs["frame_id"]

        # control
        actions = []
        for subtask in subtasks:
            action = action_generator.getAction(subtask)
            actions += action

        env.send(actions)

    env.close()
    print(f"[INFO]: Score {moneys[-1]}")

    if args.show_statistics:
        import matplotlib.pyplot as plt
        import numpy as np

        # time - money
        fig, ax = plt.subplots()
        ax.plot(moneys, label="money")
        ax.set_xlabel("time")
        ax.set_ylabel("money")
        fig.savefig("money.png")

        # task info
        task_durations = np.array(
            [task_info["duration"] for task_info in task_log])
        fig, ax = plt.subplots()
        ax.hist(task_durations, bins=100)
        ax.set_xlabel("duration")
        ax.set_ylabel("count")
        fig.savefig("task_durations.png")

        print("[INFO]: Task duration {} +- {}".format(
            np.mean(task_durations), np.std(task_durations)))

        import pickle
        pickle.dump(task_log, open("task_log.pkl", "wb"))
