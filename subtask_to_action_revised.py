import math
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np

from task_utils import Subtask, SubtaskType


# check if point p1 is between two other lines(p2, p3) and (p4, p5)
def isBetween(p1, p2, p3, p4, p5):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    x5, y5 = p5
    return (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2) >= 0 \
        and (x1 - x2) * (y4 - y2) - (y1 - y2) * (x4 - x2) <= 0 \
        and (x1 - x5) * (y3 - y5) - (y1 - y5) * (x3 - x5) >= 0 \
        and (x1 - x5) * (y4 - y5) - (y1 - y5) * (x4 - x5) <= 0


# check if two lines are parallel
def isParallel(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    return (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0

# get the crossed points of two line segments


def getCrossPoint(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    # get crossed point of two lines
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
        ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
        ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return x, y


# check if two line segments intersect
def isIntersect(p1, p2, p3, p4):
    def ccw(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) > \
            (p2[1] - p1[1]) * (p3[0] - p1[0])  # cross product

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) \
        and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class SubtaskToAction:
    def __init__(self, params: Optional[Dict] = None):
        self.Range = namedtuple("Range", ["min", "max"])
        self.Point = namedtuple("Point", ["x", "y"])
        self.Vec = namedtuple("Vec", ["x", "y"])

        self.max_line_speed = self.Range(-2, 6)
        self.max_angular_speed = self.Range(-math.pi, math.pi)

        # [lasting_frames, recorded_l_speed]
        self.collision_avoidance_stat = [[0, 0] for _ in range(4)]

        # not assigned params: use init ones
        if not params:
            self.params = {
                "collision_predict_time_1": 1.3,
                "collision_predict_time_2": 2.5,
                "avoid_collision_angular_speed": math.pi,
                "avoid_collision_l_speed_ratio": 0.5,
                "reaching_wall_threshold_1": 1,
                "reaching_wall_threshold_2": 0.3,
                "predict_scale": 0.8,
                "close_angle_difference_penalty_ratio": 5,
                "angle_difference_penalty_speed": 4,
                "same_direction_threshold": 0.52,
            }
        else:
            self.params = params

    # get action from a single subtask
    def getAction(self, subtask: Subtask, obs: Dict[str, Any]) -> List[str]:
        robot_id, robot_stat = subtask.robot_id, subtask.robot_stat
        cur_pos, cur_line_speed, cur_angular_speed, cur_theta = \
            self.Point(robot_stat['loc_x'], robot_stat['loc_y']), \
            self.Vec(robot_stat['line_speed_x'], robot_stat['line_speed_y']), \
            robot_stat['angular_speed'], robot_stat['theta']

        # radius of robot
        self_robot_radius = 0.45 if robot_stat['item_type'] == 0 else 0.53

        # deciding speed
        close_angle_difference_penalty_ratio = self.params['close_angle_difference_penalty_ratio']
        angle_difference_penalty_speed = self.params['angle_difference_penalty_speed']

        # parameters that can be tuned
        collision_predict_time = self.params["collision_predict_time_1"] if robot_stat[
            'item_type'] == 0 else self.params["collision_predict_time_2"]
        avoid_collision_angular_speed = self.params["avoid_collision_angular_speed"]
        avoid_collision_l_speed_ratio = self.params["avoid_collision_l_speed_ratio"]
        same_direction_threshold = self.params["same_direction_threshold"]

        # if not carry items: longer detect distance
        reaching_wall_threshold_1 = self.params['reaching_wall_threshold_1'] * np.sqrt(
            cur_line_speed.x ** 2 + cur_line_speed.y ** 2) / 3
        reaching_wall_threshold_2 = self.params['reaching_wall_threshold_2'] * np.sqrt(
            cur_line_speed.x ** 2 + cur_line_speed.y ** 2) / 3
        predict_scale = self.params['predict_scale']
        episilon = 1e-1
        frame_time = 0.02

        # GOTO
        if subtask.subtask_type == SubtaskType.GOTO:
            # task: get target and current moving status; output r_speed and f_speed
            target_pos = self.Point(
                subtask.station_stat['loc_x'],
                subtask.station_stat['loc_y']
            )
            l_speed, a_speed = 6, math.pi

            # predict position and angle
            predicted_theta = cur_theta + frame_time * cur_angular_speed * predict_scale
            if predicted_theta < 0:
                predicted_theta += 2 * math.pi

            predicted_x = cur_pos.x + frame_time * cur_line_speed.x * predict_scale
            predicted_y = cur_pos.y + frame_time * cur_line_speed.y * predict_scale

            predicted_spatial_distance = np.sqrt(
                (target_pos.y - cur_pos.y - frame_time * cur_line_speed.y * predict_scale) ** 2 +
                (target_pos.x - cur_pos.x - frame_time *
                 cur_line_speed.x * predict_scale) ** 2
            )
            '''
            Rotation
            '''
            # calculate angle
            target_angle = np.arctan2(
                target_pos.y - cur_pos.y, target_pos.x - cur_pos.x)

            predicted_target_angle = np.arctan2(
                target_pos.y - cur_pos.y - frame_time * cur_line_speed.y * predict_scale,
                target_pos.x - cur_pos.x - frame_time * cur_line_speed.x * predict_scale
            )

            if target_angle < 0:
                target_angle += 2 * math.pi
            if cur_theta < 0:
                cur_theta += 2 * math.pi
            if predicted_target_angle < 0:
                predicted_target_angle += 2 * math.pi

            # angle difference
            angle_difference = target_angle - cur_theta
            if angle_difference < 0:
                angle_difference += 2 * math.pi
            predicted_angle_difference = predicted_target_angle - predicted_theta
            if predicted_angle_difference < 0:
                predicted_angle_difference += 2 * math.pi

            # greedy select: direction with smaller angles
            rotate_direction = 1
            if 0 <= predicted_angle_difference <= math.pi:
                predicted_angle_to_rotate = predicted_angle_difference
            else:
                rotate_direction = -1
                predicted_angle_to_rotate = math.pi * 2 - predicted_angle_difference

            # rotate speed: linear decay with distance
            a_speed = rotate_direction * \
                min(self.max_angular_speed.max * 2 *
                    (predicted_angle_to_rotate / math.pi + 0.1), math.pi)

            '''
            Forward
            '''

            # reaching vertical edges: stop
            if (predicted_y - self_robot_radius <= reaching_wall_threshold_1 and cur_line_speed.y < - episilon) or \
                    (predicted_y + self_robot_radius >= 50 - reaching_wall_threshold_1 and cur_line_speed.y > episilon):
                # slow down linearly
                # np.sqrt(cur_line_speed.y ** 2 + cur_line_speed.x ** 2) / max(cur_line_speed.y, 1) * min(predicted_y - self_robot_radius, 50 - self_robot_radius - predicted_y)
                l_speed = 1
                a_speed = min(abs(a_speed * 5), math.pi) * rotate_direction
                # too close to wall: stop
                if (predicted_y - self_robot_radius <= reaching_wall_threshold_2 and cur_line_speed.y < - episilon) or \
                        (predicted_y + self_robot_radius >= 50 - reaching_wall_threshold_2 and cur_line_speed.y > episilon):
                    l_speed = 0.3

            # reaching horizontal edges: stop
            elif (predicted_x - self_robot_radius <= reaching_wall_threshold_1 and cur_line_speed.x < - episilon) or \
                    (predicted_x + self_robot_radius >= 50 - reaching_wall_threshold_1 and cur_line_speed.x > episilon):
                # slow down linearly
                # np.sqrt(cur_line_speed.y ** 2 + cur_line_speed.x ** 2) / max(cur_line_speed.x, 1) * min(predicted_x - self_robot_radius, 50 - self_robot_radius - predicted_x)
                l_speed = 1
                a_speed = min(abs(a_speed * 5), math.pi) * rotate_direction
                # too close to wall: stop
                if (predicted_x - self_robot_radius <= reaching_wall_threshold_2 and cur_line_speed.x < - episilon) or \
                        (predicted_x + self_robot_radius >= 50 - reaching_wall_threshold_2 and cur_line_speed.x > episilon):
                    l_speed = 0.3

            # too close
            if predicted_spatial_distance <= 5:
                normalized_difference = predicted_angle_difference \
                    if predicted_angle_difference <= math.pi \
                    else math.pi * 2 - predicted_angle_difference
                l_speed = min(l_speed,
                              self.max_line_speed.max *
                              (1 / max(normalized_difference *
                                       close_angle_difference_penalty_ratio, 1)))
            # angle difference too large: slowly forward and rotate
            elif math.pi / 2 - episilon <= predicted_angle_difference <= math.pi * 1.5 + episilon:
                l_speed = min(l_speed, angle_difference_penalty_speed)

            else:
                l_speed = 6

                # angle difference too large: slowly forward and rotate
                # if math.pi / 2 - episilon <= predicted_angle_difference <= math.pi * 1.5 + episilon:
                #     l_speed = 4
                #     # close to target: linearly slow down
                #     if predicted_spatial_distance <= 6:
                #         l_speed = self.max_line_speed.max * \
                #             (predicted_spatial_distance  / 6 / max(predicted_angle_difference, 1)) + 0.5

                # # smaller angle difference: move forward
                # else:
                #     # circling
                #     if predicted_spatial_distance <= 0.6 and abs(cur_angular_speed) >= math.pi / 8:
                #         l_speed = 0.2

                #     # too close: linearly slow down
                #     elif predicted_spatial_distance <= 3:
                #         l_speed = self.max_line_speed.max * \
                #             (predicted_spatial_distance / 3) + 3
                #     else:
                #         l_speed = self.max_line_speed.max

            '''
            Collision Avoidance
            '''

            # continue movement to avoid collision
            if self.collision_avoidance_stat[robot_id][0] > 0:
                self.collision_avoidance_stat[robot_id][0] -= 1
                # l_speed = self.collision_avoidance_stat[robot_id][1]
                a_speed = self.collision_avoidance_stat[robot_id][1]

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
                    intersect_flag = False
                    # parallel: check if line (p1, p2) is between line (p3, p4) and line (p5, p6)
                    if isParallel(self_track_lines[0][0], self_track_lines[0][1], track_lines[0][0], track_lines[0][1]):
                        if (isBetween(self_track_lines[0][0], track_lines[0][0], track_lines[0][1], track_lines[1][0], track_lines[1][1]) and
                                isBetween(self_track_lines[0][1], track_lines[0][0], track_lines[0][1], track_lines[1][0], track_lines[1][1])) or\
                            (isBetween(self_track_lines[1][0], track_lines[0][0], track_lines[0][1], track_lines[1][0], track_lines[1][1]) and
                                isBetween(self_track_lines[1][1], track_lines[0][0], track_lines[0][1], track_lines[1][0], track_lines[1][1])):
                            # check if same direction: pass
                            if np.dot(np.array([cur_line_speed.x, cur_line_speed.y]), np.array([robot['line_speed_x'], robot['line_speed_y']])) >= episilon:
                                pass
                            # opposite direction: check distance
                            elif np.sqrt((cur_pos.x - robot_pos.x) ** 2 + (cur_pos.y - robot_pos.y) ** 2) <= self_robot_radius + robot_radius or \
                                    np.sqrt((self_predicted_pos.x - robot_pos.x) ** 2 + (self_predicted_pos.y - robot_pos.y) ** 2) <= self_robot_radius + robot_radius or \
                                    np.sqrt((cur_pos.x - predicted_pos.x) ** 2 + (cur_pos.y - predicted_pos.y) ** 2) <= self_robot_radius + robot_radius:
                                # turn right
                                self.collision_avoidance_stat[robot_id][0] = 3
                                self.collision_avoidance_stat[robot_id][1] = - \
                                    avoid_collision_angular_speed
                                a_speed = - avoid_collision_angular_speed

                    else:
                        # not parallel: check if line (p1, p2) intersect with line (p3, p4)
                        for self_track_line in self_track_lines:
                            for track_line in track_lines:
                                if isIntersect(self_track_line[0], self_track_line[1], track_line[0], track_line[1]):
                                    intersect_flag = True
                                    break

                        # may collision, try to avoid
                        if intersect_flag:
                            # get crossed point
                            cross_point_x, cross_point_y = getCrossPoint(
                                (cur_pos.x, cur_pos.y),
                                (self_predicted_pos.x, self_predicted_pos.y),
                                (robot_pos.x, robot_pos.y),
                                (predicted_pos.x, predicted_pos.y)
                            )
                            cross_point = self.Point(
                                cross_point_x, cross_point_y)
                            # calculate arriving time
                            self_arriving_time = np.sqrt(
                                (cross_point.x - cur_pos.x) ** 2 +
                                (cross_point.y - cur_pos.y) ** 2
                            ) / self_robot_speed_mod
                            robot_arriving_time = np.sqrt(
                                (cross_point.x - robot_pos.x) ** 2 +
                                (cross_point.y - robot_pos.y) ** 2
                            ) / robot_speed_mod

                            # opposite direction without cross: turn
                            if np.dot(np.array([cur_line_speed.x, cur_line_speed.y]), np.array([robot['line_speed_x'], robot['line_speed_y']])) < 0:
                                # calculate angle
                                self_angle = np.arctan2(
                                    cur_line_speed.y, cur_line_speed.x)
                                if self_angle < 0:
                                    self_angle += 2 * np.pi
                                robot_angle = np.arctan2(
                                    - robot['line_speed_y'], - robot['line_speed_x'])
                                if robot_angle < 0:
                                    robot_angle += 2 * np.pi
                                angle_diff = self_angle - robot_angle
                                if angle_diff < 0:
                                    angle_diff += 2 * np.pi

                                # if angle diff is large: slow down
                                if min(angle_diff, 2*math.pi-angle_diff) > same_direction_threshold:
                                    # slow down
                                    if self_arriving_time > robot_arriving_time:
                                        # search for maximum speed
                                        min_l_speed = 0
                                        max_l_speed = self.max_line_speed.max
                                        while max_l_speed - min_l_speed > episilon:
                                            mid_l_speed = (
                                                min_l_speed + max_l_speed) / 2
                                            mid_predicted_pos = self.Point(
                                                cur_pos.x + mid_l_speed * cur_line_speed.x /
                                                self_robot_speed_mod * collision_predict_time * 1.2,
                                                cur_pos.y + mid_l_speed * cur_line_speed.y /
                                                self_robot_speed_mod * collision_predict_time * 1.2
                                            )
                                            # check distance
                                            if np.sqrt(
                                                (mid_predicted_pos.x - cross_point.x) ** 2 +
                                                (mid_predicted_pos.y -
                                                 cross_point.y) ** 2
                                            ) < robot_radius + self_robot_radius:
                                                # too close
                                                max_l_speed = mid_l_speed
                                            else:
                                                min_l_speed = mid_l_speed

                                        l_speed = min(l_speed, min_l_speed)
                                    else:
                                        pass
                                # if angle diff is small: turn
                                else:
                                    # check if the self_robot is on the right side of the line from other_robot to its predicted position
                                    self.collision_avoidance_stat[robot_id][0] = 3
                                    if np.cross(np.array([robot['line_speed_x'], robot['line_speed_y']]),
                                                np.array([cur_pos.x - robot_pos.x, cur_pos.y - robot_pos.y])) > 0:
                                        # TODO: left and right may be reversed
                                        # turn left
                                        a_speed = - avoid_collision_angular_speed
                                    else:
                                        # turn right
                                        a_speed = avoid_collision_angular_speed
                                    self.collision_avoidance_stat[robot_id][1] = a_speed

                            # same direction: robot with order smaller slow down
                            else:
                                # if another robot is by the wall
                                if np.sqrt((robot_pos.x - 0) ** 2 + (robot_pos.y - 0) ** 2) <= robot_radius or \
                                        np.sqrt((robot_pos.x - 0) ** 2 + (robot_pos.y - 50) ** 2) <= robot_radius or \
                                    np.sqrt((robot_pos.x - 50) ** 2 + (robot_pos.y - 0) ** 2) <= robot_radius or \
                                        np.sqrt((robot_pos.x - 50) ** 2 + (robot_pos.y - 50) ** 2) <= robot_radius:
                                    # if self_robot is not by the wall: stop and wait
                                    if np.sqrt((cur_pos.x - 0) ** 2 + (cur_pos.y - 0) ** 2) > self_robot_radius and \
                                            np.sqrt((cur_pos.x - 0) ** 2 + (cur_pos.y - 50) ** 2) > self_robot_radius and \
                                        np.sqrt((cur_pos.x - 50) ** 2 + (cur_pos.y - 0) ** 2) > self_robot_radius and \
                                            np.sqrt((cur_pos.x - 50) ** 2 + (cur_pos.y - 50) ** 2) > self_robot_radius:
                                        # stop
                                        l_speed = min(l_speed, 0)

                                # if robot arrives later: slow down
                                if self_arriving_time > robot_arriving_time:
                                    # search for maximum speed that avoid collision
                                    min_l_speed = 0
                                    max_l_speed = self.max_line_speed.max
                                    while max_l_speed - min_l_speed > episilon:
                                        mid_l_speed = (
                                            min_l_speed + max_l_speed) / 2
                                        mid_predicted_pos = self.Point(
                                            cur_pos.x + mid_l_speed * cur_line_speed.x /
                                            self_robot_speed_mod * collision_predict_time * 1.2,
                                            cur_pos.y + mid_l_speed * cur_line_speed.y /
                                            self_robot_speed_mod * collision_predict_time * 1.2
                                        )
                                        # check distance
                                        if np.sqrt(
                                            (mid_predicted_pos.x - cross_point.x) ** 2 +
                                            (mid_predicted_pos.y -
                                             cross_point.y) ** 2
                                        ) < robot_radius + self_robot_radius:
                                            # too close
                                            max_l_speed = mid_l_speed
                                        else:
                                            min_l_speed = mid_l_speed

                                    l_speed = min(l_speed, min_l_speed)

                # # update robot's position
                # self.moving_area[robot_id][0, self.current_window_length % self.moving_window_size], \
                #     self.moving_area[robot_id][1, self.current_window_length % self.moving_window_size] = \
                #     cur_pos.x, cur_pos.y
                # self.current_window_length += 1
                # # get the size of the area
                # if self.current_window_length >= self.moving_window_size:
                #     area_min_x, area_min_y = np.min(
                #         self.moving_area[robot_id][0]), np.min(self.moving_area[robot_id][1])
                #     area_max_x, area_max_y = np.max(
                #         self.moving_area[robot_id][0]), np.max(self.moving_area[robot_id][1])
                #     area_size = (area_max_x - area_min_x) * \
                #         (area_max_y - area_min_y)
                #     # if the area is too small, slow down
                #     if area_size < self.area_size_threshold:
                #         l_speed = min(l_speed, 0.5)

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
