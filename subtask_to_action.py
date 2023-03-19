import math
from typing import List

import numpy as np

from task_utils import Subtask, SubtaskType


class SubtaskToAction:
    def __init__(self):
        self.max_line_speed = (-2, 6)
        self.max_angular_speed = (-math.pi, math.pi)

    # get action from a single subtask
    def get_action(self, subtask: Subtask) -> List[str]:
        # robot_obs: 'station_id': -1, 'item_type': 0, 'time_coef': 0.0, 'momentum_coef': 0.0,
        # 'angular_speed': 0.0, 'line_speed': 0.0, 'theta': 0.0, 'loc_x': 0.0, 'loc_y': 24.75
        robot_id, robot_stat = subtask.robot_id, subtask.robot_stat
        cur_pos, cur_line_speed, cur_angular_speed, cur_theta = (robot_stat['loc_x'], robot_stat['loc_y']), \
            robot_stat['line_speed_x'], robot_stat['angular_speed'], \
            robot_stat['theta']
        # GOTO
        if subtask.subtask_type == SubtaskType.GOTO:
            # task: get target and current moving status; output r_speed and f_speed
            target_pos = subtask.station_stat['loc_x'], subtask.station_stat['loc_y']

            # rotation
            target_angle = np.arctan2(
                target_pos[1] - cur_pos[1], target_pos[0] - cur_pos[0])
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
                min(self.max_angular_speed[1] * 2 *
                    (angle_to_rotate / math.pi + 0.1), math.pi)
            # print(
            #     f'[Py] Current angles, target:{target_angle:.4f}, '
            #     f'robot:{cur_theta:.4f}, speed:{cur_angular_speed:.4f}'
            # )

            # forward
            # too close: linearly slow down
            if angle_difference <= 0.5 or angle_difference >= 2*math.pi - 0.5:
                spatial_distance = np.sqrt(
                    (target_pos[1] - cur_pos[1]) ** 2 + (target_pos[0] - cur_pos[0]) ** 2)
                if spatial_distance <= 5:
                    l_speed = self.max_line_speed[1] * \
                        (spatial_distance / 5 + 0.05)
                else:
                    l_speed = self.max_line_speed[1]
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
