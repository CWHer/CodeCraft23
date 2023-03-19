import argparse
import os
import random
import unittest

from RobotEnv.env_wrapper import EnvWrapper
from subtask_to_action import SubtaskToAction
from task_to_subtask import TaskManager
from task_utils import Subtask, SubtaskType, Task, TaskType


class TestAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fmt: off
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
        cls.args = args
        # fmt: on

        launch_command = f"{args.env_binary_name} {args.env_args} "\
            f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
        cls.launch_command = launch_command

    def setUp(self):
        self.env = EnvWrapper(self.args.pipe_name, self.launch_command)

    def tearDown(self):
        self.env.close()

    def testSubtask(self):
        action_generator = SubtaskToAction()
        task_manager = TaskManager()

        obs = self.env.reset()
        subtask_done = False
        while True:
            obs, done = self.env.recv()
            if done:
                break

            subtask = Subtask(
                SubtaskType.GOTO,
                item_type=None,
                robot_id=0,
                station_id=0,
            )
            subtask.update(obs)
            if task_manager.isSubtaskDone(subtask):
                subtask_done = True
                print(f"[INFO]: frame_id {obs['frame_id']}")
                break

            actions = action_generator.getAction(subtask)
            self.env.send(actions)
        self.assertTrue(subtask_done)

    def testTask(self):
        action_generator = SubtaskToAction()
        task_manager = TaskManager()

        obs = self.env.reset()
        task_done = False
        while True:
            obs, done = self.env.recv()
            if done:
                break

            task = Task(
                TaskType.BUY,
                item_type=1,
                robot_id=0,
                station_id=0,
            )
            task.update(obs)
            subtask = task_manager.makeSubtask(task, obs)
            if task_manager.isTaskDone(task, subtask):
                task_done = True
                print(f"[INFO]: frame_id {obs['frame_id']}")
                break
            if task_manager.isSubtaskDone(subtask):
                print(f"[INFO]: {subtask.subtask_type} Done")

            actions = action_generator.getAction(subtask)
            self.env.send(actions)
        self.assertTrue(task_done)


if __name__ == "__main__":
    unittest.main()
