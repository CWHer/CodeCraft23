from typing import Any, Dict, List

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

    def isTaskValid(self, task: Task, obs: Dict[str, Any]) -> bool:
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
        # 1. both robot 0 & 1 goto station 0
        # 2. not enough money
        # 3. too long waiting time
        # 4. multiple sell
        # 5. want to buy thing that won't be created
        # ...
        return tasks


    def filterInvalidTasks(self, tasks: List[List[Task]], obs: Dict[str, Any]) -> List[List[Task]]:
        # TODO: coarse filter of all tasks
        return tasks


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
    for task in tasks:
        task_checker.isTaskValid(task, obs)
    env.close()
