import random
from typing import Any, Dict, List

import numpy as np

from task_utils import MetaTask, decayFunc

from .task_manager import ItemTaskManager


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
    def __init__(self,
                 params: Dict[str, np.ndarray] = {
                     "item_delta": np.array([100, 100, 100, 300, 300, 300, 900]),
                     "station_type_delta": np.array([0, 0, 0, 0, 0, 0, 0, -50, -50]),
                     "station_input_delta": np.array([0, 0, 0, 0, 0, 0, 0, 0])
                 }
                 ) -> None:
        super().__init__()
        self.input_money = [3000, 4400, 5800, 15400, 17200, 19200, 76000]
        self.output_money = [6000, 7600, 9200, 22500, 25000, 27500, 105000]
        # self.delta = np.array(self.output_money) - np.array(self.input_money)
        self.item_delta = params["item_delta"]
        self.station_type_delta = params["station_type_delta"]
        self.station_input_delta = params["station_input_delta"]

    def priorityValue(self,
                      task: MetaTask,
                      estimated_total_time: float,
                      obs: Dict[str, Any]) -> float:
        item_value = self.item_delta[task.item_type - 1] * \
            decayFunc(estimated_total_time, 9000, 0.8)
        src_station_type = obs["stations"][task.src_station_id]["station_type"]
        dst_station_type = obs["stations"][task.dst_station_id]["station_type"]
        station_value = self.station_type_delta[dst_station_type - 1]
        input_value = 0
        for i in range(len(self.station_input_delta)):
            if (1 << i) & task.dst_input_status:
                input_value += self.station_input_delta[i]

        return item_value + station_value + input_value

    def assign(self,
               obs: Dict[str, Any],
               station_tasks: List[List[MetaTask]],
               assigned_tasks: List[List[MetaTask]]
               ) -> bool:
        # HACK: FIXME: DO NOT reschedule tasks, we avoid this by scheduling ahead
        assert len(assigned_tasks) == self.num_robots
        all_station_tasks = sum(station_tasks, [])
        idle_indices = [i for i in range(
            self.num_robots) if not assigned_tasks[i]]
        if not all_station_tasks \
                or not idle_indices:
            return False

        max_efficiency = 0
        selected_task = None
        selected_robot_id = 0

        # TODO: sell to non-sink station first
        for robot_id in idle_indices:
            robot_loc = np.array([
                obs["robots"][robot_id]["loc_x"],
                obs["robots"][robot_id]["loc_y"]
            ])
            for task in all_station_tasks:
                src_loc = np.array([
                    obs["stations"][task.src_station_id]["loc_x"],
                    obs["stations"][task.src_station_id]["loc_y"]
                ])
                robot_src_time = \
                    ItemTaskManager.moveTimeEst(robot_loc, src_loc)
                estimated_total_time = max(
                    max(robot_src_time, task.src_ready_time.max) +
                    task.dst_src_time,
                    task.dst_ready_time.max
                )
                # NOTE: ignore tasks that cannot be finished in time
                if estimated_total_time + obs["frame_id"] >= 9000:
                    continue

                delta = self.priorityValue(task, estimated_total_time, obs)
                efficiency = delta / estimated_total_time
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    selected_task = task
                    selected_robot_id = robot_id

        if not selected_task:
            return False

        selected_task.robot_id = selected_robot_id
        selected_task.update(obs)
        assigned_tasks[selected_robot_id].append(selected_task)
        return True
