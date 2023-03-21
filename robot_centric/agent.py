from typing import Dict, List

from subtask_to_action import SubtaskToAction
from task_to_subtask import TaskHelper

from .scheduler import BaseScheduler
from .task_manager import TaskManager


class RobotBasedAgent:
    def __init__(self, scheduler: BaseScheduler) -> None:
        self.num_robots = 4
        self.moneys, self.task_log = [], []
        self.last_obs = [None for _ in range(self.num_robots)]

        self.scheduler = scheduler
        self.task_manager = TaskManager()
        self.task_helper = TaskHelper()
        self.subtask_to_action = SubtaskToAction()

    def reset(self):
        self.moneys, self.task_log = [], []
        self.last_obs = [None for _ in range(self.num_robots)]
        self.scheduler.clear(list(range(self.num_robots)))

    def step(self, obs: Dict) -> List[str]:
        assert obs is not None
        self.moneys.append(obs["money"])

        # check task status
        subtasks = [None] * self.num_robots
        for i, task in enumerate(self.scheduler.stat()):
            if task is None:
                continue
            task.update(obs)
            subtask = self.task_helper.makeSubtask(task, obs)
            if self.task_helper.isTaskDone(task, subtask):
                self.task_log.append({
                    "start_time": self.last_obs[i]["frame_id"],
                    "end_time": obs["frame_id"],
                    "duration": obs["frame_id"] - self.last_obs[i]["frame_id"],
                    "task_info": task,
                    "obs_info": self.last_obs[i],
                })
                continue
            subtasks[i] = subtask

        # make decision
        robot_indices = [
            i for i in range(self.num_robots)
            if subtasks[i] is None
        ]  # idle robots
        self.scheduler.clear(robot_indices)
        candidate_tasks = self.task_manager.genTasks(obs)
        candidate_tasks = self.task_manager.filterInvalidTasks(
            candidate_tasks, obs)
        for i in range(self.num_robots):
            if subtasks[i] is None:
                robot_tasks = self.task_manager.checkConflict(
                    candidate_tasks[i], self.scheduler.stat(), obs)
                selected_task = self.scheduler.select(i, robot_tasks, obs)
                subtask = self.task_helper.makeSubtask(selected_task, obs)
                subtasks[i] = subtask
                self.last_obs[i] = obs

        # control
        actions = []
        for subtask in subtasks:
            assert subtask is not None
            action = self.subtask_to_action.getAction(subtask, obs)
            actions += action

        return actions

    def showStatistics(self) -> Dict:
        print(f"[INFO]: Score {self.moneys[-1]}")
        # show unfinished:
        for i, task in enumerate(self.scheduler.stat()):
            if task is not None:
                print(
                    f"[INFO]: Unfinished - Robot {i}, task_type {task.task_type}, "
                    f"station {task.station_id}, item {task.item_type}"
                )

        import matplotlib.pyplot as plt
        import numpy as np

        # time - money
        fig, ax = plt.subplots()
        ax.plot(self.moneys, label="money")
        ax.set_xlabel("time")
        ax.set_ylabel("money")
        fig.savefig("money.png")

        # task info
        task_durations = np.array(
            [task_info["duration"] for task_info in self.task_log])
        fig, ax = plt.subplots()
        ax.hist(task_durations, bins=100)
        ax.set_xlabel("duration")
        ax.set_ylabel("count")
        fig.savefig("task_durations.png")

        print("[INFO]: Task duration {:.2f} +- {:.2f}".format(
            np.mean(task_durations), np.std(task_durations)))

        import datetime
        import pickle
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        pickle.dump(self.task_log, open(f"stat_{now}.pkl", "wb"))

        return {
            "money": self.moneys,
            "task_log": self.task_log,
        }
