from typing import Dict, List

from task_to_subtask import TaskHelper

from .scheduler import BaseScheduler
from .task_manager import ItemTaskManager


class ItemBasedAgent:
    def __init__(self,
                 scheduler: BaseScheduler,
                 use_revised_control=False,
                 movement_params=None) -> None:
        self.num_robots = 4
        self.reset()

        self.scheduler = scheduler
        self.task_manager = ItemTaskManager()
        self.task_helper = TaskHelper()
        if not use_revised_control:
            from subtask_to_action import SubtaskToAction
            self.subtask_to_action = SubtaskToAction(movement_params)
        else:
            from subtask_to_action_revised import SubtaskToAction
            self.subtask_to_action = SubtaskToAction(movement_params)

    def reset(self):
        self.last_obs = None
        self.moneys, self.task_log = [], []
        self.last_frame = [0 for _ in range(self.num_robots)]
        self.assigned_tasks = [[] for _ in range(self.num_robots)]

    def step(self, obs: Dict) -> List[str]:
        assert obs is not None
        self.last_obs = obs
        self.moneys.append(obs["money"])

        # check task status
        for i, tasks in enumerate(self.assigned_tasks):
            while tasks:
                current_task = tasks[0]
                current_task.update(obs)
                if self.task_helper.isMetaTaskDone(current_task):
                    self.task_log.append({
                        "start_time": self.last_frame[i],
                        "end_time": obs["frame_id"],
                        "duration": obs["frame_id"] - self.last_frame[i],
                        "task_info": current_task,
                    })
                    tasks.pop(0)
                else:
                    break

        # make decision
        # HACK: FIXME: DO NOT reschedule tasks, we avoid this by scheduling ahead
        idle_indices = [
            i for i in range(self.num_robots)
            if not self.assigned_tasks[i]
        ]
        for _ in range(len(idle_indices)):
            candidate_tasks = \
                self.task_manager.genTasks(
                    obs, self.assigned_tasks)
            result = self.scheduler.assign(
                obs, candidate_tasks, self.assigned_tasks)
            if not result:
                break
        for index in idle_indices:
            if self.assigned_tasks[index]:
                self.last_frame[index] = obs["frame_id"]
            else:
                print(
                    f"[INFO]: Robot {index} is idle "
                    f"at frame {obs['frame_id']}"
                )
        # control
        actions = [" "]
        for meta_tasks in self.assigned_tasks:
            if meta_tasks:
                task = self.task_helper.makeTask(meta_tasks[0], obs)
                subtask = self.task_helper.makeSubtask(task, obs)
                action = self.subtask_to_action.getAction(subtask, obs)
                actions += action

        return actions

    def showStatistics(self) -> Dict:
        print(f"[INFO]: Score {self.moneys[-1]}")
        print(f"[INFO]: Max Score {max(self.moneys)}")

        task_durations = [
            task_info["duration"]
            for task_info in self.task_log
        ]
        for i, meta_tasks in enumerate(self.assigned_tasks):
            for task in meta_tasks:
                print(
                    f"[INFO]: Unfinished - Robot {i}, item {task.item_type}, "
                    f"src station {task.src_station_id}, dst station {task.dst_station_id}"
                )
            if meta_tasks:
                task_durations.append(
                    self.last_obs["frame_id"] - self.last_frame[i])

        import matplotlib.pyplot as plt
        import numpy as np

        # time - money
        fig, ax = plt.subplots()
        ax.plot(self.moneys, label="money")
        ax.set_xlabel("time")
        ax.set_ylabel("money")
        fig.savefig("money.png")

        # task info
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


class ReplayAgent:
    def __init__(self,
                 assigned_tasks: List[List],
                 movement_params=None) -> None:
        self.num_robots = 4
        self.moneys = []
        self.assigned_tasks = assigned_tasks

        self.task_helper = TaskHelper()
        self.subtask_to_action = SubtaskToAction(movement_params)

    def step(self, obs: Dict) -> List[str]:
        assert obs is not None
        self.last_obs = obs
        self.moneys.append(obs["money"])

        # check task status
        for i, tasks in enumerate(self.assigned_tasks):
            while tasks:
                current_task = tasks[0]
                current_task.update(obs)
                if self.task_helper.isMetaTaskDone(current_task):
                    tasks.pop(0)
                else:
                    break

        # make decision
        # HACK: FIXME: DO NOT reschedule tasks, we avoid this by scheduling ahead
        idle_indices = [
            i for i in range(self.num_robots)
            if not self.assigned_tasks[i]
        ]
        for index in idle_indices:
            print(
                f"[INFO]: Robot {index} is idle "
                f"at frame {obs['frame_id']}"
            )
        # control
        actions = [" "]
        for meta_tasks in self.assigned_tasks:
            if meta_tasks:
                task = self.task_helper.makeTask(meta_tasks[0], obs)
                subtask = self.task_helper.makeSubtask(task, obs)
                action = self.subtask_to_action.getAction(subtask, obs)
                actions += action

        return actions
