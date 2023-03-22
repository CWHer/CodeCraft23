
import random
from typing import Any, Dict, List

from task_utils import MetaTask


class BaseScheduler:
    def __init__(self) -> None:
        self.num_robots = 4

    def assign(self,
               obs: Dict[str, Any],
               station_tasks: List[List[MetaTask]],
               assigned_tasks: List[List[MetaTask]]
               ) -> None:
        assert len(assigned_tasks) == self.num_robots
        all_station_tasks = sum(station_tasks, [])
        for i, tasks in enumerate(assigned_tasks):
            if not tasks and all_station_tasks:
                index = random.randint(0, len(all_station_tasks) - 1)
                selected_task = all_station_tasks.pop(index)
                selected_task.robot_id = i
                selected_task.update(obs)
                tasks.append(selected_task)


class GreedyScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__()

    def assign(self,
               obs: Dict[str, Any],
               station_tasks: List[List[MetaTask]],
               assigned_tasks: List[List[MetaTask]]
               ) -> None:
        # HACK: FIXME: DO NOT reschedule tasks, we avoid this by scheduling ahead
        # TODO:
        raise NotImplementedError()
