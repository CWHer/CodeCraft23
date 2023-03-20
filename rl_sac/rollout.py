import os
import random
from typing import Any, List, Tuple

import numpy as np
from Env.env_utils import obsToNumpy, taskToNumpy
from Env.RobotEnv.env_wrapper import EnvWrapper
from Env.subtask_to_action import SubtaskToAction
from Env.task_checker import TaskChecker
from Env.task_to_subtask import TaskManager
from scheduler import BaseScheduler
from utils import Transition


class RolloutWorker:
    def __init__(self,
                 reward_scale=0.001,
                 map_id="maps/1.txt",
                 env_binary_name="./Robot_fast",
                 env_args="-d",
                 env_wrapper_name="./env_wrapper",
                 pipe_name=f"/tmp/pipe_{random.random()}",
                 ) -> None:
        self.reward_scale = reward_scale

        self.robot_env_path = os.path.join(
            os.path.dirname(__file__), "Env", "RobotEnv")
        map_id = os.path.join(self.robot_env_path, map_id)
        env_binary_name = os.path.join(self.robot_env_path, env_binary_name)
        env_wrapper_name = os.path.join(self.robot_env_path, env_wrapper_name)

        launch_command = f"{env_binary_name} {env_args} "\
            f"-m {map_id} \"{env_wrapper_name} {pipe_name}\""
        self.env = EnvWrapper(pipe_name, launch_command)

        self.action_generator = SubtaskToAction()
        self.task_manager = TaskManager()
        self.task_checker = TaskChecker()

        self.num_robots = 4
        self.total_frame = 9000

        with open(map_id, "r") as f:
            env_map = ''.join(f.readlines())
            self.num_station = 0
            for c in env_map:
                self.num_station += c.isdigit()

        self.obs_padding = 2 + 6 * self.num_station + 10 * self.num_robots
        self.task_padding = 4 + 10 + 6

    def rollout(self, scheduler: BaseScheduler) \
            -> Tuple[List[Transition], List[int], List[Any]]:
        """
        Return:
            episode: List of transition
            money: List of money at each frame
            task_log: List of task dict
        """

        # statistics
        moneys, task_log = [], []
        last_obs = [None for _ in range(self.num_robots)]
        all_tasks = [[] for _ in range(self.num_robots)]

        env_map = self.env.reset()
        while True:
            obs, done = self.env.recv()
            if done:
                break
            assert obs is not None
            moneys.append(obs["money"])

            # check task status
            subtasks = [None] * self.num_robots
            for i, task in enumerate(scheduler.stat()):
                if task is None:
                    continue
                task.update(obs)
                subtask = self.task_manager.makeSubtask(task, obs)
                if self.task_manager.isTaskDone(task, subtask):
                    task_log.append({
                        "start_time": last_obs[i]["frame_id"],
                        "end_time": obs["frame_id"],
                        "duration": obs["frame_id"] - last_obs[i]["frame_id"],
                        "action": task,
                        "candidata_actions": all_tasks[i],
                        # HACK: only last frame changes money
                        "reward": moneys[-1] - moneys[-2],
                        "obs": last_obs[i],
                    })
                    continue
                subtasks[i] = subtask

            # make decision
            robot_indices = [
                i for i in range(self.num_robots)
                if subtasks[i] is None
            ]  # idle robots
            scheduler.clear(robot_indices)
            candidate_tasks = self.task_checker.genTasks(obs)
            candidate_tasks = self.task_checker.filterInvalidTasks(
                candidate_tasks, obs)
            for i in range(self.num_robots):
                if subtasks[i] is None:
                    robot_tasks = self.task_checker.checkConflict(
                        candidate_tasks[i], scheduler.stat(), obs)
                    selected_task = scheduler.select(i, robot_tasks, obs)
                    subtask = self.task_manager.makeSubtask(selected_task, obs)
                    subtasks[i] = subtask
                    last_obs[i], all_tasks[i] = obs, robot_tasks

            # control
            actions = []
            for subtask in subtasks:
                action = self.action_generator.getAction(subtask)
                actions += action

            self.env.send(actions)

        print("[INFO]: Rollout finished")
        print("[INFO]: generating episode...")

        # HACK: FIXME: we assume at most one task is done at each frame
        end_time = [log["end_time"] for log in task_log]
        if not task_log or len(end_time) != len(set(end_time)):
            raise RuntimeError("Multiple tasks are done at the same frame")

        episode = []
        task_log.sort(key=lambda x: x["end_time"])
        for i in range(len(task_log) - 1):
            episode.append(Transition(
                obs=obsToNumpy(task_log[i]["obs"], self.obs_padding),
                action=taskToNumpy(task_log[i]["action"], self.task_padding),
                candidate_actions=np.array([
                    taskToNumpy(task, self.task_padding)
                    for task in task_log[i]["candidata_actions"]
                ]),
                reward=task_log[i]["reward"] * self.reward_scale,
                next_obs=obsToNumpy(task_log[i + 1]["obs"], self.obs_padding),
                done=False,
            ))
        episode.append(Transition(
            obs=obsToNumpy(task_log[-1]["obs"], self.obs_padding),
            action=taskToNumpy(task_log[-1]["action"], self.task_padding),
            candidate_actions=np.array([
                taskToNumpy(task, self.task_padding)
                for task in task_log[-1]["candidata_actions"]
            ]),
            reward=task_log[-1]["reward"] * self.reward_scale,
            next_obs=obsToNumpy(None, self.obs_padding),
            done=True,
        ))

        return episode, moneys, task_log


if __name__ == "__main__":
    import time
    random_scheduler = BaseScheduler()
    rollout_worker = RolloutWorker()
    start_time = time.time()
    episode, *_ = rollout_worker.rollout(random_scheduler)
    duration = time.time() - start_time
    print("[INFO]: time elapsed {:.2f}s".format(duration))

    from replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(1000)
    buffer.add(episode)
    batch = buffer.sample(4)
