from typing import Any, Dict

from task_utils import Subtask, SubtaskType, Task, TaskType


class TaskManager:
    def makeSubtask(self, task: Task, obs: Dict[str, Any]) -> Subtask:
        # NOTE: HACK: we assume all tasks are valid
        if task.task_type == TaskType.BUY:
            subtask = Subtask(
                SubtaskType.GOTO,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=task.station_id
            )
            subtask.update(obs)
            if self.isSubtaskDone(subtask):
                subtask.subtask_type = SubtaskType.BUY
            return subtask
        elif task.task_type == TaskType.SELL:
            subtask = Subtask(
                SubtaskType.GOTO,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=task.station_id
            )
            subtask.update(obs)
            if self.isSubtaskDone(subtask):
                subtask.subtask_type = SubtaskType.SELL
            return subtask
        elif task.task_type == TaskType.DESTROY:
            subtask = Subtask(
                SubtaskType.DESTROY,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=None,
            )
            subtask.update(obs)
            return subtask
        else:
            raise NotImplementedError()

    def isTaskDone(self, task: Task, subtask: Subtask) -> bool:
        return task.task_type == subtask.subtask_type \
            and self.isSubtaskDone(subtask)

    def isSubtaskDone(self, subtask: Subtask) -> bool:
        if subtask.subtask_type == SubtaskType.GOTO:
            return subtask.robot_stat["station_id"] == subtask.station_id
        elif subtask.subtask_type == SubtaskType.BUY:
            return subtask.robot_stat["item_type"] == subtask.item_type
        elif subtask.subtask_type == SubtaskType.SELL:
            return subtask.robot_stat["item_type"] == 0
        elif subtask.subtask_type == SubtaskType.DESTROY:
            return subtask.robot_stat["item_type"] == 0
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    # test
    task_manager = TaskManager()

    task = Task(
        TaskType.BUY,
        item_type=1,
        robot_id=0,
        station_id=0,
    )
    obs = {
        "robots": [
            {
                "station_id": -1,
            }
        ],
        "stations": [{}]
    }
    subtask = task_manager.makeSubtask(task, obs)
    assert subtask.subtask_type == SubtaskType.GOTO

    obs = {
        "robots": [
            {
                "station_id": 0,
                "item_type": 1
            }
        ],
        "stations": [{}]
    }
    subtask = task_manager.makeSubtask(task, obs)
    assert subtask.subtask_type == SubtaskType.BUY
    assert task_manager.isSubtaskDone(subtask)
