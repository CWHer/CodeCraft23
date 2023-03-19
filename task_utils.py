import dataclasses
import enum
from typing import Any, Dict, Optional


class TaskType(enum.Enum):
    BUY = 1
    SELL = 2
    DESTROY = 3


@dataclasses.dataclass
class Task:
    task_type: TaskType
    item_type: int
    robot_id: int
    # robot_stat = {
    #     'station_id': -1,
    #     'item_type': 0,
    #     'time_coef': 0.0,
    #     'momentum_coef': 0.0,
    #     'angular_speed': 0.0,
    #     'line_speed': 0.0,
    #     'theta': 0.0,
    #     'loc_x': 0.0,
    #     'loc_y': 24.75
    # }
    robot_stat: Dict[str, Any]
    station_id: Optional[int]
    # station_stat = {
    #     'station_type': 1,
    #     'loc_x': 1.25,
    #     'loc_y': 48.75,
    #     'remain_time': 49,
    #     'input_status': 0,
    #     'output_status': 0
    # }
    station_stat: Optional[Dict[str, Any]]


class SubtaskType(enum.Enum):
    GOTO = 0
    BUY = 1
    SELL = 2
    DESTROY = 3


@dataclasses.dataclass
class Subtask:
    subtask_type: SubtaskType
    item_type: Optional[int]
    robot_id: int
    robot_stat: Dict[str, Any]
    station_id: Optional[int]
    station_stat: Optional[Dict[str, Any]]