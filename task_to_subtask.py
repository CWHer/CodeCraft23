from typing import Any, Dict

from task_utils import MetaTask, Subtask, SubtaskType, Task, TaskType


class TaskHelper:
    def makeTask(self, meta_task: MetaTask, obs: Dict[str, Any]) -> Task:
        meta_task.reach_src |= meta_task.robot_stat["station_id"] == meta_task.src_station_id
        meta_task.reach_dst |= meta_task.robot_stat["station_id"] == meta_task.dst_station_id
        meta_task.owned_item |= meta_task.robot_stat["item_type"] == meta_task.item_type

        if not meta_task.owned_item:
            task = Task(
                TaskType.BUY,
                reach_dst=meta_task.reach_src,
                item_type=meta_task.item_type,
                robot_id=meta_task.robot_id,
                station_id=meta_task.src_station_id
            )
        else:
            task = Task(
                TaskType.SELL,
                reach_dst=meta_task.reach_dst,
                item_type=meta_task.item_type,
                robot_id=meta_task.robot_id,
                station_id=meta_task.dst_station_id
            )

        task.update(obs)
        return task

    def makeSubtask(self, task: Task, obs: Dict[str, Any]) -> Subtask:
        # NOTE: HACK: we assume all tasks are valid
        if task.task_type == TaskType.BUY:
            task.reach_dst |= task.robot_stat["station_id"] == task.station_id
            subtask = Subtask(
                SubtaskType.GOTO if not task.reach_dst else SubtaskType.BUY,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=task.station_id
            )
        elif task.task_type == TaskType.SELL:
            task.reach_dst |= task.robot_stat["station_id"] == task.station_id
            subtask = Subtask(
                SubtaskType.GOTO if not task.reach_dst else SubtaskType.SELL,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=task.station_id
            )
        elif task.task_type == TaskType.DESTROY:
            subtask = Subtask(
                SubtaskType.DESTROY,
                item_type=task.item_type,
                robot_id=task.robot_id,
                station_id=None,
            )
        else:
            raise NotImplementedError()

        subtask.update(obs)
        return subtask

    def isMetaTaskDone(self, meta_task: MetaTask) -> bool:
        return meta_task.reach_src and \
            meta_task.reach_dst and meta_task.owned_item and \
            meta_task.robot_stat["item_type"] == 0

    def isTaskDone(self, task: Task) -> bool:
        if task.task_type == TaskType.BUY:
            return task.reach_dst and \
                task.robot_stat["item_type"] == task.item_type
        elif task.task_type == TaskType.SELL:
            return task.reach_dst and \
                task.robot_stat["item_type"] == 0
        elif task.task_type == TaskType.DESTROY:
            return task.robot_stat["item_type"] == 0
        else:
            raise NotImplementedError()

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
    task_manager = TaskHelper()

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
