import gym
import argparse
import os
import sys
import torch

import pandas as pd
from gym import spaces
import numpy as np
from so_abalone_env import PredatorPrey 
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, ggi, ifr, ifrnum):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PredatorPrey(out_csv_name='results/sb3_reward_dqn{}'.format(rank), ggi=ggi, iFR=ifr, iFRnum=ifrnum)
        
        return env
    return _init


if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Deep-Q-Learning""")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-b", dest="buffer", type=int, default=10000, required=False, help="Buffer size of DQN.\n")
    prs.add_argument("-ef", dest="exploration_f", type=float, default=0.2, required=False, help="exploration_fraction.\n")
    prs.add_argument("-efs", dest="exploration_f_e", type=float, default=0.1, required=False, help="exploration_final_eps.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=128, required=False, help="Batch size for NN.\n")
    prs.add_argument("-targfr", dest="target_net_freq", type=int, default=500, required=False, help="Update freq for target.\n")
    prs.add_argument("-fr", dest="ifr", type=int, default=2, required=False, help="Functional Response for SC\n")
    prs.add_argument("-fnum", dest="ifrnum", type=int, default=2, required=False, help="Functional Response Num for SC\n")
    prs.add_argument("-w", dest="weight", type=int, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_true", default=False, help="Run GGI algo or not.\n")
    args = prs.parse_args()

    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(f'ggi{i}' if ggi else i, ggi, args.ifr, args.ifrnum) for i in range(n_cpu)])
    reward_space = 2
    
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "optimizer_kwargs": dict(
            weight_decay = 0.001,
            eps = 1e-5,
        )
    }

    # create the model
    if ggi:
        from stable_baselines3.dqn_ggi import DQN_GGI, GGIMlpPolicy
        model = DQN_GGI(
            policy=GGIMlpPolicy,
            env=env,
            reward_space=reward_space,
            weight_coef=args.weight,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            train_freq = 50,
            gradient_steps = 30,
            policy_kwargs = policy_kwargs,
            verbose=1,
            learning_starts = args.target_net_freq,
            target_update_interval = args.target_net_freq,
        )

    else:
        from stable_baselines3.dqn import DQN, MlpPolicy
        model = DQN(
            policy=MlpPolicy,
            env=env,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            train_freq = 50,
            gradient_steps = 30,
            policy_kwargs = policy_kwargs,
            verbose=1,
            learning_starts = args.target_net_freq,
            target_update_interval = args.target_net_freq,
        )

    # learning of the model
    model.learn(total_timesteps=1000000)
