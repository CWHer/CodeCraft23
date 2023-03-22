
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
               ) -> bool:
        assert len(assigned_tasks) == self.num_robots
        all_station_tasks = sum(station_tasks, [])
        idle_indices = [i for i in range(
            self.num_robots) if not assigned_tasks[i]]
        if not all_station_tasks \
                or not idle_indices:
            return False
        index = random.randint(0, len(all_station_tasks) - 1)
        selected_task = all_station_tasks[index]
        robot_id = random.choice(idle_indices)
        selected_task.robot_id = robot_id
        selected_task.update(obs)
        assigned_tasks[robot_id].append(selected_task)
        return True


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
