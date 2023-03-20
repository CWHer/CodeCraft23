from typing import Dict, Optional

import numpy as np

from .task_utils import Task


def obsToNumpy(obs: Optional[Dict], padding=0) -> np.ndarray:
    if obs is None:
        return -np.ones(shape=(padding, ), dtype=np.float32)
    arr = [[obs["frame_id"], obs["money"]]]
    if "stations" in obs:
        for station in obs["stations"]:
            arr.append(list(station.values()))
    if "robots" in obs:
        for robot in obs["robots"]:
            arr.append(list(robot.values()))
    arr = np.concatenate(arr, dtype=np.float32)
    arr = np.pad(
        arr, (0, padding - arr.shape[0]),
        "constant", constant_values=-1
    )
    return arr


def taskToNumpy(task: Optional[Task], padding=0) -> np.ndarray:
    if task is None:
        return -np.ones(shape=(padding, ), dtype=np.float32)

    def toInt(x): return x if x else -1
    arr = [[task.task_type.value, toInt(task.item_type),
            task.robot_id, toInt(task.station_id)]]
    arr.append(list(task.robot_stat.values()))
    if task.station_stat is not None:
        arr.append(list(task.station_stat.values()))
    arr = np.concatenate(arr, dtype=np.float32)
    arr = np.pad(
        arr, (0, padding - arr.shape[0]),
        "constant", constant_values=-1
    )
    return arr
