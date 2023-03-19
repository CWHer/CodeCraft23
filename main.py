import argparse
import os
import random

from RobotEnv.env_wrapper import EnvWrapper
from scheduler import DummyScheduler
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
    scheduler = DummyScheduler()

    obs = env.reset()
    final_reward = 0
    while True:
        obs, done = env.recv()
        if done:
            break
        assert obs is not None
        final_reward = obs["money"]

        # check task status
        subtasks = [None] * num_robots
        for i, task in enumerate(scheduler.stat()):
            if task is None:
                continue
            task.update(obs)
            subtask = task_manager.makeSubtask(task, obs)
            if not task_manager.isTaskDone(task, subtask):
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

        # control
        actions = []
        for subtask in subtasks:
            action = action_generator.getAction(subtask)
            actions += action

        env.send(actions)

    print(f"[INFO]: Reward {final_reward}")
    env.close()
