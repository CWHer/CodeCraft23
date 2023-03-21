from typing import Any, Dict, List

import numpy as np

from task_utils import MetaTask, TimeRange


class ItemTaskManager:
    def __init__(self) -> None:
        self.item_types = [1, 2, 3, 4, 5, 6, 7]
        self.station_specs = {
            1: {
                "input": [],
                "output": [1]
            },
            2: {
                "input": [],
                "output": [2]
            },
            3: {
                "input": [],
                "output": [3]
            },
            4: {
                "input": [1, 2],
                "output": [4]
            },
            5: {
                "input": [1, 3],
                "output": [5]
            },
            6: {
                "input": [2, 3],
                "output": [6]
            },
            7: {
                "input": [4, 5, 6],
                "output": [7]
            },
            8: {
                "input": [7],
                "output": []
            },
            9: {
                "input": [1, 2, 3, 4, 5, 6, 7],
                "output": []
            },
        }

    @staticmethod
    def moveTimeEst(src: np.ndarray, dst: np.ndarray) -> float:
        max_speed = 6
        scale_factor = 1.1 / max_speed
        return scale_factor * \
            np.linalg.norm(np.array(src) - np.array(dst)).item()

    @staticmethod
    def betweenStation(src: Dict, dst: Dict) -> float:
        return ItemTaskManager.moveTimeEst(
            np.array([src["loc_x"], src["loc_y"]]),
            np.array([dst["loc_x"], dst["loc_y"]])
        )

    def genTasks(self,
                 obs: Dict[str, Any],
                 assigned_tasks: List[List[MetaTask]]
                 ) -> List[List[MetaTask]]:
        # TODO: consider both obs and assigned tasks
        # NOTE: naive esimation,
        #   e.g., src station definitely can produce the item (producing/done)
        # TODO: one stage forward estimation
        #   e.g., src station may lack some input items, which can definitely be produced
        # TODO: recursive estimation
        #   e.g., src station may lack some input items, and input of input stations also lack some items, ...

        tasks_per_item = []
        for item_type in self.item_types:
            item_tasks = []
            for i, src_station in enumerate(obs["stations"]):
                src_type = src_station["station_type"]
                if not item_type in \
                        self.station_specs[src_type]["output"]:
                    continue
                for j, dst_station in enumerate(obs["stations"]):
                    dst_type = dst_station["station_type"]
                    if not item_type in \
                            self.station_specs[dst_type]["input"]:
                        continue

                    # FIXME: naive filtering
                    if src_station["remain_time"] == -1 \
                            and src_station["output_status"] == 0:
                        continue
                    src_ready_time = TimeRange(0, 0) \
                        if src_station["output_status"] == 1 \
                        else TimeRange(src_station["remain_time"], src_station["remain_time"])
                    if dst_station["input_status"] & (1 << item_type) \
                            and dst_station["remain_time"] >= 0 \
                            and dst_station["output_status"] == 1:
                        continue
                    dst_ready_time = TimeRange(0, 0) \
                        if not (dst_station["input_status"] & (1 << item_type)) or dst_station["output_status"] == 1 \
                        else TimeRange(dst_station["remain_time"], dst_station["remain_time"])

                    meta_task = MetaTask(
                        item_type=item_type,
                        src_station_id=i,
                        dst_station_id=j,
                        src_ready_time=src_ready_time,
                        dst_ready_time=dst_ready_time,
                        dst_src_time=self.betweenStation(
                            src_station, dst_station),
                    )

                    item_tasks.append(meta_task)
            tasks_per_item.append(item_tasks)

        return tasks_per_item
