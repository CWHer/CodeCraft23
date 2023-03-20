from collections import namedtuple
from typing import Any, Dict, List

Transition = namedtuple(
    "Transition",
    [
        "obs",
        "action",
        "candidate_actions",
        "reward",
        "done",
    ],
)


def concatDict(dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Concatenate a list of dictionaries into a dictionary of lists.

    :return: A dictionary of lists.
    """
    result: Dict[str, List[Any]] = {}
    for d in dicts:
        for k, v in d.items():
            if k not in result:
                result[k] = []
            result[k].append(v)
    return result
