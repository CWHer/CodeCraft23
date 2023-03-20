import random
from typing import Any, Dict, List, Optional

from task_utils import Task, TaskType


class BaseScheduler:
    def __init__(self) -> None:
        random.seed(1927)
        self.num_robots = 4
        self.assigned_tasks: List[Optional[Task]] = [None] * self.num_robots

    def select(self,
               robot_id: int,
               tasks: List[Task],
               obs: Dict[str, Any]
               ) -> Task:
        selected_task = random.choice(tasks)
        self.assigned_tasks[robot_id] = selected_task
        return selected_task

    def clear(self, indices: List[int]):
        for index in indices:
            self.assigned_tasks[index] = None

    def stat(self) -> List[Optional[Task]]:
        return self.assigned_tasks


class GreedyScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__()

    def select(self,
               robot_id: int,
               tasks: List[Task],
               obs: Dict[str, Any]
               ) -> Task:
        robot_pos_x, robot_pos_y = \
            obs['robots'][robot_id]['loc_x'], obs['robots'][robot_id]['loc_y']

        destroy_task = None
        # check buy / sell: robot carry items
        if obs['robots'][robot_id]['item_type'] != 0:
            work_mode = 'sell'
            dist = float('inf')
            selected_task = None
        else:
            work_mode = 'buy'
            dist_by_item = [float('inf') for _ in range(7)]
            selected_task = [None for _ in range(7)]

        # find the nearest station
        for task in tasks:
            # destroy
            if task.task_type == TaskType.DESTROY:
                destroy_task = task
            # sell: nearest station
            elif work_mode == 'sell':
                station_pos_x = obs['stations'][task.station_id]['loc_x']
                station_pos_y = obs['stations'][task.station_id]['loc_y']
                if abs(robot_pos_x - station_pos_x) + \
                        abs(robot_pos_y - station_pos_y) < dist:
                    dist = abs(robot_pos_x - station_pos_x) + \
                        abs(robot_pos_y - station_pos_y)
                    selected_task = task
            # buy: nearest station for each item; random choose one
            elif work_mode == 'buy':
                station_pos_x, station_pos_y = obs['stations'][task.station_id][
                    'loc_x'], obs['stations'][task.station_id]['loc_y']
                if abs(robot_pos_x - station_pos_x) + \
                        abs(robot_pos_y - station_pos_y) < dist_by_item[task.item_type-1]:
                    dist_by_item[task.item_type-1] = abs(robot_pos_x - station_pos_x) + \
                        abs(robot_pos_y - station_pos_y)
                    selected_task[task.item_type-1] = task

        # for buy: random choose one item
        if work_mode == 'buy':
            selected_task = random.choice(
                [i for i in selected_task if i is not None])
        elif work_mode == 'sell' and selected_task is None:
            selected_task = destroy_task
        self.assigned_tasks[robot_id] = selected_task
        return selected_task
