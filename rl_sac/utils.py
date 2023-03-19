import time
from typing import Any, Dict, List, Union


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


def gatherInfo(
    start_time: float,
    train_collector: Any,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting transitions in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (env_step per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (env_step per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = max(0, time.time() - start_time)
    model_time = duration
    result: Dict[str, Union[float, str]] = {
        "duration": f"{duration:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
    }

    assert train_collector is not None
    collect_step, collect_episode, collect_time = train_collector.getStat()
    model_time = max(0, model_time - collect_time)
    train_speed = collect_step / duration
    result.update(
        {
            "train_step": collect_step,
            "train_episode": collect_episode,
            "train_time/collector": f"{collect_time:.2f}s",
            "train_time/model": f"{model_time:.2f}s",
            "train_speed": f"{train_speed:.2f} step/s",
        }
    )
    return result
