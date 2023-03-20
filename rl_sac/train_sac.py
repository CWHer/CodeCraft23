import argparse
import datetime
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import tqdm
from replay_buffer import ReplayBuffer
from rollout import RolloutWorker
from sac_network import MLP, ActorNet, CrticNet
from sac_policy import DiscreteSACPolicy
from scheduler import RLScheduler
from torch.utils.tensorboard import SummaryWriter


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--critic-lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--n-episode", type=int, default=10)
    parser.add_argument("--n-update", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-env-thread", type=int, default=24)
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()


def rolloutFunc(scheduler):
    with torch.no_grad():
        pid = os.getpid()
        random.seed(pid)
        np.random.seed(pid)
        torch.manual_seed(pid)
        rollout_worker = RolloutWorker(pipe_name=f"/tmp/pipe_{pid}")
        result = rollout_worker.rollout(scheduler)
        if result is None:
            print("[ERROR] RuntimeError: ", result)
        rollout_worker.close()
        return result


def trainSAC(args):
    # fix seed
    # HACK: this would slow down the training
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # collector
    dummy_rollout_worker = RolloutWorker()

    obs_encoder = MLP([128, 64])
    act_encoder = MLP([64, 64])
    actor = ActorNet(obs_encoder, act_encoder)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = CrticNet(obs_encoder, act_encoder, MLP([128, 64, 1]))
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = CrticNet(obs_encoder, act_encoder, MLP([128, 64, 1]))
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        dummy_rollout_worker.obs_padding,
        dummy_rollout_worker.task_padding,
    )

    rl_scheduler = RLScheduler(policy)

    # replay buffer
    buffer = ReplayBuffer(args.buffer_size)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "sac"
    log_name = os.path.join(args.algo_name, now)
    log_path = os.path.join(os.getcwd(), args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    episode_count = 0
    update_count = 0
    for i in tqdm.trange(args.epoch):
        policy.eval()
        with ProcessPoolExecutor(max_workers=args.num_env_thread) as executor:
            for result in executor.map(
                    rolloutFunc, [rl_scheduler] * args.n_episode):
                if result is None:
                    continue
                eposide, moneys, task_log = result
                buffer.add(eposide)
                episode_count += 1
                writer.add_scalar("reward", moneys[-1], episode_count)

        policy.train()
        for _ in range(args.n_update):
            batch = buffer.sample(args.batch_size)
            result = policy.update(batch)

            # fmt: off
            update_count += 1
            writer.add_scalar("loss/actor", result["loss/actor"], update_count)
            writer.add_scalar("loss/critic1", result["loss/critic1"], update_count)
            writer.add_scalar("loss/critic2", result["loss/critic2"], update_count)
            # fmt: on
            print(result)


if __name__ == "__main__":
    trainSAC(parseArgs())
