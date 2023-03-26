from typing import Any, Dict, List

from task_to_subtask import TaskHelper


class ReplayAgent:
    def __init__(self,
                 assigned_tasks: List[List],
                 subtask_to_action: Any,
                 movement_params=None) -> None:
        self.num_robots = 4
        self.moneys = []
        self.assigned_tasks = assigned_tasks

        self.task_helper = TaskHelper()
        self.subtask_to_action = subtask_to_action

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
            pass
            # print(
            #     f"[INFO]: Robot {index} is idle "
            #     f"at frame {obs['frame_id']}"
            # )
        # control
        actions = [" "]
        for meta_tasks in self.assigned_tasks:
            if meta_tasks:
                task = self.task_helper.makeTask(meta_tasks[0], obs)
                subtask = self.task_helper.makeSubtask(task, obs)
                action = self.subtask_to_action.getAction(subtask, obs)
                actions += action

        return actions
