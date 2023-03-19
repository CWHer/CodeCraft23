from typing import Any, Dict, List, Optional

from task_utils import Task


class DummyScheduler:
    def __init__(self) -> None:
        self.num_robots = 4
        self.assigned_tasks: List[Optional[Task]] = [None] * self.num_robots

    def select(self,
               robot_id: int,
               tasks: List[Task],
               obs: Dict[str, Any]
               ) -> Task:
        selected_task = tasks[0]
        self.assigned_tasks[robot_id] = selected_task
        return selected_task

    def clear(self, indices: List[int]):
        for index in indices:
            self.assigned_tasks[index] = None

    def stat(self) -> List[Optional[Task]]:
        return self.assigned_tasks
