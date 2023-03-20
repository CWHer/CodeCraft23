import math
from collections import namedtuple
from typing import List

import numpy as np

from .task_utils import Subtask, SubtaskType


class SubtaskToAction:
    def __init__(self):
        self.Range = namedtuple("Range", ["min", "max"])
        self.Point = namedtuple("Point", ["x", "y"])
        self.Vec = namedtuple("Vec", ["x", "y"])

        self.max_line_speed = self.Range(-2, 6)
        self.max_angular_speed = self.Range(-math.pi, math.pi)

    # get action from a single subtask
    def getAction(self, subtask: Subtask) -> List[str]:
        robot_id, robot_stat = subtask.robot_id, subtask.robot_stat
        cur_pos, cur_line_speed, cur_angular_speed, cur_theta = \
            self.Point(robot_stat['loc_x'], robot_stat['loc_y']), \
            self.Vec(robot_stat['line_speed_x'], robot_stat['line_speed_y']), \
            robot_stat['angular_speed'], robot_stat['theta']
        # GOTO
        if subtask.subtask_type == SubtaskType.GOTO:
            # task: get target and current moving status; output r_speed and f_speed
            target_pos = self.Point(
                subtask.station_stat['loc_x'],
                subtask.station_stat['loc_y']
            )

            # rotation
            target_angle = np.arctan2(
                target_pos.y - cur_pos.y, target_pos.x - cur_pos.x)
            if target_angle < 0:
                target_angle += 2 * math.pi
            if cur_theta < 0:
                cur_theta += 2 * math.pi

            # greedy select: direction with smaller angles
            rotate_direction, angle_to_rotate = 1, 0
            angle_difference = target_angle-cur_theta
            if angle_difference < 0:
                angle_difference += 2 * math.pi

            if 0 <= angle_difference <= math.pi:
                angle_to_rotate = angle_difference
            else:
                rotate_direction = -1
                angle_to_rotate = math.pi * 2 - angle_difference

            # rotate speed: linear decay with distance
            a_speed = rotate_direction * \
                min(self.max_angular_speed.max * 2 *
                    (angle_to_rotate / math.pi + 0.1), math.pi)
            # print(
            #     f'[Py] Current angles, target:{target_angle:.4f}, '
            #     f'robot:{cur_theta:.4f}, speed:{cur_angular_speed:.4f}'
            # )

            # forward
            # too close: linearly slow down
            if angle_difference <= 0.5 or angle_difference >= 2 * math.pi - 0.5:
                spatial_distance = np.sqrt(
                    (target_pos.y - cur_pos.y) ** 2 + (target_pos.x - cur_pos.x) ** 2)
                if spatial_distance <= 5:
                    l_speed = self.max_line_speed.max * \
                        (spatial_distance / 5 + 0.05)
                else:
                    l_speed = self.max_line_speed.max
            # angle difference too large: stop and rotate
            else:
                l_speed = 0

            return [f'forward {robot_id} {l_speed}', f'rotate {robot_id} {a_speed}']

        # BUY
        elif subtask.subtask_type == SubtaskType.BUY:
            return [f'buy {robot_id}']

        # SELL
        elif subtask.subtask_type == SubtaskType.SELL:
            return [f'sell {robot_id}']

        # DESTROY
        elif subtask.subtask_type == SubtaskType.DESTROY:
            return [f'destroy {robot_id}']

        else:
            raise NotImplementedError()
