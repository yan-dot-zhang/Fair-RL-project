import gym
import argparse
import os
import sys

import pandas as pd
from gym import spaces
import numpy as np
from sympy import im
from so_abalone_env import PredatorPrey 

from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, ggi, ifr, ifrnum):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PredatorPrey(out_csv_name='results/sb3_reward_ppo{}'.format(rank), ggi=ggi, iFR=ifr, iFRnum=ifrnum)
        
        return env
    return _init

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""PPO""")
    prs.add_argument("-gam", dest="gamma", type=float, default=0.99, required=False, help="discount factor of PPO.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.00005, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.1, required=False, help="clip_range of PPO.\n")
    prs.add_argument("-st", dest="steps", type=int, default=128, required=False, help="n steps for PPO.\n")
    prs.add_argument("-fr", dest="ifr", type=int, default=2, required=False, help="Functional Response for SC\n")
    prs.add_argument("-fnum", dest="ifrnum", type=int, default=2, required=False, help="Functional Response Num for SC\n")
    prs.add_argument("-w", dest="weight", type=int or float, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_true", default=False, help="Run GGI algo or not.\n")
    args = prs.parse_args()

    # multiprocess environment
    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(f'ggi{i}' if ggi else i, ggi, args.ifr, args.ifrnum) for i in range(n_cpu)])
    reward_space = 2

    if ggi:
        from stable_baselines3.ppo_ggi import PPO_GGI, GGIMlpPolicy
        model = PPO_GGI(
            policy = GGIMlpPolicy,
            env = env,
            reward_space = reward_space,
            weight_coef = args.weight,
            gamma = args.gamma,
            n_steps = args.steps,
            verbose = 1,
            ent_coef = 0.01,
            vf_coef = 0.25,
            learning_rate = args.alpha,
            clip_range = args.clip_range,
            normalize_advantage = False,
        )
    else:
        from stable_baselines3.ppo import PPO, MlpPolicy
        model = PPO(
            policy = MlpPolicy,
            env = env,
            gamma = args.gamma,
            n_steps = args.steps,
            verbose = 1,
            ent_coef = 0.01,
            vf_coef = 0.25,
            learning_rate = args.alpha,
            clip_range = args.clip_range,
            normalize_advantage = False,
        )

    model.learn(total_timesteps=1000000)
