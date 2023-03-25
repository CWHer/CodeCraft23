from typing import Any, Dict, List, Tuple

from task_utils import MetaTask, TimeRange


class ItemTaskManager:
    def __init__(self) -> None:
        self.item_types = [1, 2, 3, 4, 5, 6, 7]
        self.sink_stations = [8, 9]
        self.source_stations = [1, 2, 3]
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
        for k, v in self.station_specs.items():
            input_stat = 0
            for item_type in v["input"]:
                input_stat += 1 << item_type
            v["full"] = input_stat
        self.item_specs = {
            item_type: {
                "src_type": [],
                "dst_type": [],
                "src_id": [],
                "dst_id": [],
            }
            for item_type in self.item_types
        }
        for k, v in self.station_specs.items():
            for item_type in v["input"]:
                self.item_specs[item_type]["dst_type"].append(k)
            for item_type in v["output"]:
                self.item_specs[item_type]["src_type"].append(k)
        self.station_by_types = None

    @staticmethod
    def moveTimeEst(src: Tuple, dst: Tuple) -> float:
        max_speed = 6
        scale_factor = 1.1 / max_speed
        return scale_factor * \
            ((src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2) ** 0.5

    @staticmethod
    def betweenStation(src: Dict, dst: Dict) -> float:
        return ItemTaskManager.moveTimeEst(
            (src["loc_x"], src["loc_y"]),
            (dst["loc_x"], dst["loc_y"])
        )

    def currentTaskStat(self, assigned_tasks: List[List[MetaTask]]) -> Dict[int, Any]:
        reserved_stat = {
            i: {"input": [], "output": []}
            for i in range(self.num_stations)
        }
        # TODO: add estimated time
        for tasks in assigned_tasks:
            for task in tasks:
                # fmt: off
                if not task.owned_item:
                    reserved_stat[task.src_station_id]["output"].append(task.item_type)
                reserved_stat[task.dst_station_id]["input"].append(task.item_type)
                # fmt: on
        return reserved_stat

    def genTasks(self,
                 obs: Dict[str, Any],
                 assigned_tasks: List[List[MetaTask]]
                 ) -> List[List[MetaTask]]:
        # NOTE: naive esimation,
        #   e.g., src station definitely can produce the item (producing/done)
        # TODO: one stage forward estimation
        #   e.g., src station may lack some input items, which can definitely be produced
        # TODO: recursive estimation
        #   e.g., src station may lack some input items, and input of input stations also lack some items, ...

        if self.station_by_types is None:
            self.num_stations = len(obs["stations"])
            self.station_by_types = {
                station_type: []
                for station_type in self.station_specs.keys()
            }
            for i, station in enumerate(obs["stations"]):
                self.station_by_types[station["station_type"]].append(i)
            for item_type in self.item_specs:
                for station_type in self.item_specs[item_type]["src_type"]:
                    self.item_specs[item_type]["src_id"] += self.station_by_types[station_type]
                for station_type in self.item_specs[item_type]["dst_type"]:
                    self.item_specs[item_type]["dst_id"] += self.station_by_types[station_type]

        reserved_stat = self.currentTaskStat(assigned_tasks)

        tasks_per_item = []
        for item_type in self.item_types:
            item_tasks = []
            for i in self.item_specs[item_type]["src_id"]:
                for j in self.item_specs[item_type]["dst_id"]:
                    src_station = obs["stations"][i]
                    dst_station = obs["stations"][j]

                    # TODO: conflict with reserved tasks
                    if item_type in reserved_stat[i]["output"] \
                            or item_type in reserved_stat[j]["input"]:
                        continue

                    # naive filtering
                    if src_station["remain_time"] == -1 \
                            and src_station["output_status"] == 0:
                        continue
                    src_ready_time = TimeRange(0, 0) \
                        if src_station["output_status"] == 1 \
                        else TimeRange(src_station["remain_time"], src_station["remain_time"])
                    # 1. lack this item
                    # 2. input full but producing not done
                    full_stat = self.station_specs[dst_station["station_type"]]["full"]
                    if dst_station["input_status"] & (1 << item_type) \
                            and dst_station["input_status"] != full_stat:
                        continue
                    if dst_station["input_status"] == full_stat \
                            and dst_station["remain_time"] >= 0 \
                            and dst_station["output_status"] == 1:
                        continue
                    dst_ready_time = TimeRange(0, 0) \
                        if not (dst_station["input_status"] & (1 << item_type)) or dst_station["output_status"] == 1 \
                        else TimeRange(dst_station["remain_time"], dst_station["remain_time"])

                    # NOTE: record dst station input status
                    dst_input_status = dst_station["input_status"]
                    for k in reserved_stat[j]["input"]:
                        dst_input_status |= 1 << k
                    meta_task = MetaTask(
                        item_type=item_type,
                        src_station_id=i,
                        dst_station_id=j,
                        src_ready_time=src_ready_time,
                        dst_ready_time=dst_ready_time,
                        dst_src_time=self.betweenStation(
                            src_station, dst_station),
                        dst_input_status=dst_input_status,
                    )

                    item_tasks.append(meta_task)
            tasks_per_item.append(item_tasks)

        return tasks_per_item
