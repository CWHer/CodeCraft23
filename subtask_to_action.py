import math
from collections import namedtuple
from typing import List, Dict, Any

import numpy as np

from task_utils import Subtask, SubtaskType


# check if two line segments intersect
def isIntersect(p1, p2, p3, p4):
    def ccw(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) > \
            (p2[1] - p1[1]) * (p3[0] - p1[0])  # cross product

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) \
        and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class SubtaskToAction:
    def __init__(self):
        self.Range = namedtuple("Range", ["min", "max"])
        self.Point = namedtuple("Point", ["x", "y"])
        self.Vec = namedtuple("Vec", ["x", "y"])

        self.max_line_speed = self.Range(-2, 6)
        self.max_angular_speed = self.Range(-math.pi, math.pi)

    # get action from a single subtask
    def getAction(self, subtask: Subtask, obs: Dict[str, Any]) -> List[str]:
        robot_id, robot_stat = subtask.robot_id, subtask.robot_stat
        cur_pos, cur_line_speed, cur_angular_speed, cur_theta = \
            self.Point(robot_stat['loc_x'], robot_stat['loc_y']), \
            self.Vec(robot_stat['line_speed_x'], robot_stat['line_speed_y']), \
            robot_stat['angular_speed'], robot_stat['theta']
        # parameters that can be tuned
        collision_predict_time = 1.8
        reaching_wall_threshold = 0.4
        episilon = 1e-1

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

            # forward
            # reaching edge: stop
            if cur_pos.x <= reaching_wall_threshold or cur_pos.x >= 50 - reaching_wall_threshold or \
                    cur_pos.y <= reaching_wall_threshold or cur_pos.y >= 50 - reaching_wall_threshold:
                l_speed = 1
            else:
                # too close: linearly slow down
                if angle_difference <= math.pi / 2 or angle_difference >= 2 * math.pi - math.pi / 2:
                    spatial_distance = np.sqrt(
                        (target_pos.y - cur_pos.y) ** 2 + (target_pos.x - cur_pos.x) ** 2)
                    if spatial_distance <= 3:
                        l_speed = self.max_line_speed.max * \
                            (spatial_distance / 3) + 0.5
                    else:
                        l_speed = self.max_line_speed.max
                # angle difference too large: stop and rotate
                else:
                    l_speed = 2

            # detect collision
            self_robot_speed_mod = np.sqrt(
                cur_line_speed.x ** 2 + cur_line_speed.y ** 2)
            if self_robot_speed_mod >= episilon:
                # check robot to decide action: draw moving line within collision_predict_time
                self_predicted_pos = self.Point(
                    cur_pos.x + cur_line_speed.x * collision_predict_time,
                    cur_pos.y + cur_line_speed.y * collision_predict_time
                )
                # consider robot's radius
                self_robot_radius = 0.45 if robot_stat['item_type'] == 0 else 0.53
                self_vertical_addition = self.Vec(
                    cur_line_speed.y / self_robot_speed_mod * self_robot_radius, -cur_line_speed.x / self_robot_speed_mod * self_robot_radius)
                # track lines: two lines
                self_track_lines = [((cur_pos.x + self_vertical_addition.x, cur_pos.y + self_vertical_addition.y),
                                    (self_predicted_pos.x + self_vertical_addition.x, self_predicted_pos.y + self_vertical_addition.y)),
                                    ((cur_pos.x - self_vertical_addition.x, cur_pos.y - self_vertical_addition.y),
                                    (self_predicted_pos.x - self_vertical_addition.x, self_predicted_pos.y - self_vertical_addition.y))]

                # check every robot
                for robot_idx, robot in enumerate(obs['robots']):
                    if robot_idx == robot_id:
                        continue
                    # predict position
                    robot_pos = self.Point(robot['loc_x'], robot['loc_y'])
                    robot_speed = self.Vec(
                        robot['line_speed_x'], robot['line_speed_y'])
                    robot_speed_mod = np.sqrt(
                        robot_speed.x ** 2 + robot_speed.y ** 2)
                    # too slow: no need to check
                    if robot_speed_mod < episilon:
                        continue
                    predicted_pos = self.Point(
                        robot_pos.x + robot_speed.x * collision_predict_time,
                        robot_pos.y + robot_speed.y * collision_predict_time
                    )
                    # consider robot's radius
                    robot_radius = 0.45 if robot['item_type'] == 0 else 0.53
                    vertical_addition = self.Vec(
                        robot_speed.y / robot_speed_mod * robot_radius, -robot_speed.x / robot_speed_mod * robot_radius)
                    # track lines: two lines
                    track_lines = [((robot_pos.x + vertical_addition.x, robot_pos.y + vertical_addition.y),
                                    (predicted_pos.x + vertical_addition.x, predicted_pos.y + vertical_addition.y)),
                                   ((robot_pos.x - vertical_addition.x, robot_pos.y - vertical_addition.y),
                                    (predicted_pos.x - vertical_addition.x, predicted_pos.y - vertical_addition.y))]

                    # check collision
                    for self_track_line in self_track_lines:
                        for track_line in track_lines:
                            if isIntersect(self_track_line[0], self_track_line[1], track_line[0], track_line[1]):
                                # if collision, try to avoid
                                # slow down
                                # l_speed = min(l_speed, 2)
                                # opposite direction
                                if np.dot(np.array([cur_line_speed.x, cur_line_speed.y]), np.array([robot['line_speed_x'], robot['line_speed_y']])) < 0:
                                    # turn right
                                    a_speed = self.max_angular_speed.min
                                break

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
