import math
from typing import Any, Dict, List, Optional

from task_utils import Task, TaskType


class TaskChecker:
    def __init__(self) -> None:
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

        self.costs = [3000, 4400, 5800, 15400, 17200, 19200, 76000]

    def isTaskValid(self, task: Task, obs: Dict[str, Any]) -> None:
        # FIXME: this may be useless, genTasks is enough
        if task.task_type == TaskType.BUY:
            assert task.robot_stat["item_type"] == 0
            station_type = obs["stations"][task.station_id]["station_type"]
            assert task.item_type in self.station_specs[station_type]["output"]
        elif task.task_type == TaskType.SELL:
            assert task.robot_stat["item_type"] != 0 \
                and task.robot_stat["item_type"] == task.item_type
            station_type = obs["stations"][task.station_id]["station_type"]
            assert task.item_type in self.station_specs[station_type]["input"]
        elif task.task_type == TaskType.DESTROY:
            assert task.robot_stat["item_type"] != 0 \
                and task.robot_stat["item_type"] == task.item_type

    def genTasks(self, obs: Dict[str, Any]) -> List[List[Task]]:
        num_robots = len(obs["robots"])
        num_stations = len(obs["stations"])
        buy_tasks, sell_tasks, destroy_tasks = [[] for _ in range(num_robots)], \
            [[] for _ in range(num_robots)], [[] for _ in range(num_robots)]

        for i in range(num_robots):
            # destory
            if obs["robots"][i]["item_type"] != 0:
                task = Task(
                    task_type=TaskType.DESTROY,
                    item_type=obs["robots"][i]["item_type"],
                    robot_id=i,
                    station_id=None
                )
                task.update(obs)
                destroy_tasks[i].append(task)

            for j in range(num_stations):
                # buy
                station_type = obs["stations"][j]["station_type"]
                if obs["robots"][i]["item_type"] == 0:
                    for k in self.station_specs[station_type]["output"]:
                        task = Task(
                            task_type=TaskType.BUY,
                            item_type=k,
                            robot_id=i,
                            station_id=j
                        )
                        task.update(obs)
                        buy_tasks[i].append(task)
                # sell
                if obs["robots"][i]["item_type"] \
                        in self.station_specs[station_type]["input"]:
                    task = Task(
                        task_type=TaskType.SELL,
                        item_type=obs["robots"][i]["item_type"],
                        robot_id=i,
                        station_id=j
                    )
                    task.update(obs)
                    sell_tasks[i].append(task)

        tasks_by_robot = []
        for i in range(num_robots):
            tasks_by_robot.append(
                buy_tasks[i] + sell_tasks[i] + destroy_tasks[i])
        return tasks_by_robot

    def checkConflict(self,
                      tasks: List[Task],
                      assigned_tasks: List[Optional[Task]],
                      obs: Dict[str, Any]
                      ) -> List[Task]:
        # TODO: e.g., this could be important
        valid_index = [1 for _ in range(len(tasks))]

        # parameters that can be tuned
        predict_scale = 1.3
        conflict_time_threshold = 1
        conflict_robot_threshold = 1

        # predict time for assigned tasks
        assigned_predicted_time = [100000 for _ in range(4)]
        for idx, a_task in enumerate(assigned_tasks):
            if not a_task or a_task.task_type == TaskType.DESTROY:
                continue
            robot_x, robot_y = obs["robots"][a_task.robot_id]["loc_x"], obs["robots"][a_task.robot_id]["loc_y"]
            assigned_predicted_time[idx] = math.sqrt((robot_x - a_task.station_stat["loc_x"]) ** 2 + (
                robot_y - a_task.station_stat["loc_y"]) ** 2) / 6 * predict_scale

        # # calculate linear equation track for each assigned task
        # line_equations = [None for _ in range(4)]
        # for idx, a_task in enumerate(assigned_tasks):
        #     if not a_task or a_task.task_type == TaskType.DESTROY:
        #         continue
        #     else:
        #         # get points
        #         point1 = (obs["robots"][a_task.robot_id]["loc_x"], obs["robots"][a_task.robot_id]["loc_y"])
        #         point2 = (a_task.station_stat["loc_x"], a_task.station_stat["loc_y"])
        #         # calc Ax + By + C = 0
        #         A = point1[1] - point2[1]
        #         B = point2[0] - point1[0]
        #         C = point1[0] * point2[1] - point2[0] * point1[1]
        #         if abs(A) < episilon:
        #             A = 0
        #         if abs(B) < episilon:
        #             B = 0
        #         line_equations[idx] = (A, B, C)

        # check for each task
        for idx, task in enumerate(tasks):
            station_type = None if task.station_id is None \
                else task.station_stat["station_type"]

            if task.task_type == TaskType.BUY or task.task_type == TaskType.SELL:
                # # avoid collision : currently do not care about direction
                # calc Ax + By + C = 0
                # point1 = (obs["robots"][task.robot_id]["loc_x"], obs["robots"][task.robot_id]["loc_y"])
                # point2 = (task.station_stat["loc_x"], task.station_stat["loc_y"])
                # A = point1[1] - point2[1]
                # B = point2[0] - point1[0]
                # C = point1[0] * point2[1] - point2[0] * point1[1]
                # if abs(A) < episilon:
                #     A = 0
                # if abs(B) < episilon:
                #     B = 0
                # # check conflict
                # for assigned_idx, assigned_task in enumerate(assigned_tasks):
                #     if not assigned_task or a_task.task_type == TaskType.DESTROY:
                #         continue
                #     elif assigned_task.task_type == TaskType.BUY or assigned_task.task_type == TaskType.SELL:
                #         # check if the angel between two lines is less than episilon
                #         # calc the angel between two lines
                #         angle = math.acos((A * line_equations[assigned_idx][0] + B * line_equations[assigned_idx][1]) /
                #                           (math.sqrt(A ** 2 + B ** 2) * math.sqrt(line_equations[assigned_idx][0] ** 2 + line_equations[assigned_idx][1] ** 2)))

                # too many robots go to the same station: may cause bumping into each other
                robot_x, robot_y = obs["robots"][task.robot_id]["loc_x"], obs["robots"][task.robot_id]["loc_y"]
                self_predicted_time = math.sqrt((robot_x - task.station_stat["loc_x"]) ** 2 + (
                    robot_y - task.station_stat["loc_y"]) ** 2) / 6 * predict_scale
                # count conflict robots
                conflict_robot_num = 0
                for assigned_idx, assigned_task in enumerate(assigned_tasks):
                    if not assigned_task or assigned_task.task_type == TaskType.DESTROY:
                        continue
                    elif assigned_task.task_type == TaskType.BUY or assigned_task.task_type == TaskType.SELL:
                        if task.station_id == assigned_task.station_id and abs(self_predicted_time - assigned_predicted_time[assigned_idx]) < conflict_time_threshold:
                            conflict_robot_num += 1
                if conflict_robot_num >= conflict_robot_threshold:
                    valid_index[idx] = 0
                    continue

            # selling same product
            if task.task_type == TaskType.SELL and 4 <= station_type <= 7:
                for assigned_task in assigned_tasks:
                    if not assigned_task or assigned_task.task_type == TaskType.DESTROY:
                        continue
                    elif assigned_task.task_type == TaskType.SELL and task.station_id == assigned_task.station_id and task.item_type == assigned_task.item_type:
                        valid_index[idx] = 0

            # buying same processed product
            if task.task_type == TaskType.BUY and 4 <= station_type <= 7:
                for assigned_task in assigned_tasks:
                    if not assigned_task or assigned_task.task_type == TaskType.DESTROY:
                        continue
                    elif assigned_task.task_type == TaskType.BUY and task.station_id == assigned_task.station_id and task.item_type == assigned_task.item_type \
                            and task.station_stat["output_status"] != 0:
                        valid_index[idx] = 0
        # delete
        new_tasks = [task for idx, task in enumerate(
            tasks) if valid_index[idx] == 1]
        return new_tasks
        # 1. both robot 0 & 1 goto station 0
        # 2. not enough money
        # 3. too long waiting time
        # 4. multiple sell
        # 5. want to buy thing that won't be created
        # 6. some items cannot be sold if there are not receiving stations
        # ...

    def filterInvalidTasks(self, tasks: List[List[Task]], obs: Dict[str, Any]) -> List[List[Task]]:
        # TODO: coarse filter of all tasks
        # params that can be adjusted
        predict_scale = 1.3

        # stats
        new_tasks = [[] for _ in range(4)]

        # get invalid items to buy: stop buying items which has been fully placed
        # valid_items = set()
        # for station in obs['stations']:
        #     # buy
        #     station_type = station["station_type"]
        #     if 4 <= station_type <= 7:
        #         for item in self.station_specs[station_type]["output"]:
        #             if item not in valid_items:
        #                 valid_items.add(item)
        #     if obs["robots"][i]["item_type"] == 0:

        # check for individual robot
        for i in range(4):
            valid_index = [1 for _ in range(len(tasks[i]))]
            for idx, task in enumerate(tasks[i]):
                station_type = None if task.station_id is None \
                    else task.station_stat["station_type"]

                # buy product
                if task.task_type == TaskType.BUY:
                    # from processing stations
                    # not producing now
                    if 4 <= station_type <= 7 and \
                            obs['stations'][task.station_id]['remain_time'] == -1:
                        valid_index[idx] = 0

                    # not ready within predicted arriving time
                    elif 4 <= station_type <= 7 and \
                            obs['stations'][task.station_id]['output_status'] == 0:
                        # clac time
                        station_pos_x, station_pos_y = \
                            obs['stations'][task.station_id]['loc_x'], \
                            obs['stations'][task.station_id]['loc_y']
                        robot_pos_x, robot_pos_y = \
                            obs['robots'][i]['loc_x'], obs['robots'][i]['loc_y']
                        predicted_time = math.sqrt(
                            (robot_pos_x - station_pos_x) ** 2 +
                            (robot_pos_y - station_pos_y) ** 2
                        ) / 6 * predict_scale
                        # check
                        if predicted_time < obs['stations'][task.station_id]['remain_time']:
                            valid_index[idx] = 0

                # sell product
                elif task.task_type == TaskType.SELL:
                    # full station; not producing or not ready within arrival time
                    # fixme: may not be the right way to interpret; other robots may clear the station
                    if 4 <= station_type <= 7:
                        # get input stat
                        input_status = obs['stations'][task.station_id]['input_status']
                        placed_item_list = [0 for _ in range(6)]
                        for item_idx in range(6):
                            input_status //= 2
                            placed_item_list[item_idx] = input_status % 2

                        # print('Deciding:', f'robot {i} station {task.station_id} item {task.item_type} station_type {station_type}',
                        #     f'remain_time {obs["stations"][task.station_id]["remain_time"]}, item_list', placed_item_list)

                        # already_have + not producing / conjesture
                        if placed_item_list[task.item_type-1] == 1 \
                                and obs['stations'][task.station_id]['remain_time'] <= 0:
                            valid_index[idx] = 0

                        # do not have + producing / conjesture + lack one item
                        # elif placed_item_list[task.item_type-1] == 0 and obs['stations'][task.station_id]['remain_time'] >= 0 and \
                        #         sum(placed_item_list) == len(self.station_specs[station_type]["input"])-1 and obs['stations'][task.station_id]['output_status'] == 1:
                        #     valid_index[idx] = 0

                        # already_have + not ready within arrival time
                        elif placed_item_list[task.item_type-1] == 1:
                            # calc time
                            station_pos_x, station_pos_y = \
                                obs['stations'][task.station_id]['loc_x'], \
                                obs['stations'][task.station_id]['loc_y']
                            robot_pos_x, robot_pos_y = \
                                obs['robots'][i]['loc_x'], obs['robots'][i]['loc_y']
                            predicted_time = math.sqrt(
                                (robot_pos_x - station_pos_x) ** 2 +
                                (robot_pos_y - station_pos_y) ** 2
                            ) / 6 * predict_scale
                            # check
                            if predicted_time < obs['stations'][task.station_id]['remain_time']:
                                valid_index[idx] = 0

            # delete
            new_tasks[i] = [task for idx, task in enumerate(
                tasks[i]) if valid_index[idx] == 1]
        return new_tasks


if __name__ == "__main__":
    import argparse
    import os
    import random

    from RobotEnv.env_wrapper import EnvWrapper

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default="maps/1.txt")
    parser.add_argument("--env-binary-name", default="./Robot_fast")
    parser.add_argument("--env-args", default="-d")
    parser.add_argument("--env-wrapper-name", default="./env_wrapper")
    parser.add_argument("--pipe-name", default=f"/tmp/pipe_{random.random()}")
    args = parser.parse_args()
    robot_env_path = os.path.join(os.path.dirname(__file__), "RobotEnv")
    args.map_id = os.path.join(robot_env_path, args.map_id)
    args.env_binary_name = os.path.join(robot_env_path, args.env_binary_name)
    args.env_wrapper_name = os.path.join(robot_env_path, args.env_wrapper_name)
    print(args)

    launch_command = f"{args.env_binary_name} {args.env_args} "\
        f"-m {args.map_id} \"{args.env_wrapper_name} {args.pipe_name}\""
    env = EnvWrapper(args.pipe_name, launch_command)

    task_checker = TaskChecker()

    obs = env.reset()
    obs, done = env.recv()
    tasks = task_checker.genTasks(obs)
    for robot_tasks in tasks:
        for task in robot_tasks:
            task_checker.isTaskValid(task, obs)
    env.close()
    print("[INFO]: Done!")
