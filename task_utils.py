import dataclasses
import enum
from collections import namedtuple
from typing import Any, Dict, Optional

TimeRange = namedtuple("TimeRange", ["min", "max"])


def decayFunc(x, max_x, min_ratio) -> float:
    if x >= max_x:
        return min_ratio
    return (1 - (1 - (1 - x / max_x) ** 2) ** 0.5) * (1 - min_ratio) + min_ratio


@dataclasses.dataclass
class MetaTask:
    # NOTE: actually only one task type is needed, that is BUY and SELL,
    #   and BUY and DESTROY is useless
    item_type: int
    src_station_id: int
    dst_station_id: int

    # FIXME: maybe a distribution is more accurate
    src_ready_time: TimeRange
    dst_ready_time: TimeRange
    dst_src_time: float  # robot goto dst from src
    # robot_src_time: float = float("inf")  # robot goto src

    # NOTE:
    # estimated_total_time = max(
    #     max(robot_src_time, src_ready_time) + dst_src_time,
    #     dst_ready_time
    # )
    owned_item: bool = False

    dst_input_status: int = 0

    robot_id: int = -1
    robot_stat: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def update(self, obs: Dict[str, Any]):
        self.robot_stat = obs["robots"][self.robot_id]


class TaskType(enum.Enum):
    BUY = 1
    SELL = 2
    DESTROY = 3  # HACK: this action is not rational


@dataclasses.dataclass
class Task:
    task_type: TaskType
    item_type: Optional[int]
    robot_id: int
    station_id: Optional[int]
    # robot_stat = {
    #     'station_id': -1,
    #     'item_type': 0,
    #     'time_coef': 0.0,
    #     'momentum_coef': 0.0,
    #     'angular_speed': 0.0,
    #     'line_speed_x': 0.0,
    #     'line_speed_y': 0.0,
    #     'theta': 0.0,
    #     'loc_x': 24.75,
    #     'loc_y': 38.75
    # }
    robot_stat: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # station_stat = {
    #     'station_type': 1,
    #     'loc_x': 1.25,
    #     'loc_y': 48.75,
    #     'remain_time': 49,
    #     'input_status': 0,
    #     'output_status': 0
    # }
    station_stat: Optional[Dict[str, Any]] = None

    def update(self, obs: Dict[str, Any]):
        self.robot_stat = obs["robots"][self.robot_id]
        if self.station_id is not None:
            self.station_stat = obs["stations"][self.station_id]


class SubtaskType(enum.Enum):
    # NOTE: subtask is atomic
    GOTO = 0
    BUY = 1
    SELL = 2
    DESTROY = 3


@dataclasses.dataclass
class Subtask:
    subtask_type: SubtaskType
    item_type: Optional[int]
    robot_id: int
    station_id: Optional[int]
    robot_stat: Dict[str, Any] = dataclasses.field(default_factory=dict)
    station_stat: Optional[Dict[str, Any]] = None

    def update(self, obs: Dict[str, Any]):
        self.robot_stat = obs["robots"][self.robot_id]
        if self.station_id is not None:
            self.station_stat = obs["stations"][self.station_id]
