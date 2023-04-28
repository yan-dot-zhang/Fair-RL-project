import gym
import argparse
import os
import sys

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
    prs.add_argument("-a", dest="alpha", type=float, default=0.001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-b", dest="buffer", type=int, default=50000, required=False, help="Buffer size of DQN.\n")
    prs.add_argument("-ef", dest="exploration_f", type=float, default=0.1, required=False, help="exploration_fraction.\n")
    prs.add_argument("-efs", dest="exploration_f_e", type=float, default=0.1, required=False, help="exploration_final_eps.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=32, required=False, help="Batch size for NN.\n")
    prs.add_argument("-targfr", dest="target_net_freq", type=int, default=50, required=False, help="Update freq for target.\n")
    prs.add_argument("-fr", dest="ifr", type=int, default=2, required=False, help="Functional Response for SC\n")
    prs.add_argument("-fnum", dest="ifrnum", type=int, default=2, required=False, help="Functional Response Num for SC\n")
    prs.add_argument("-w", dest="weight", type=int, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_true", default=False, help="Run GGI algo or not.\n")
    prs.add_argument("-id", dest="run_index", type=int, default=0, help="Run index.\n")
    args = prs.parse_args()

    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(f'ggi{i}' if ggi else i, ggi, args.ifr, args.ifrnum) for i in range(n_cpu)])
    weight_coef = args.weight
    
    # create the model
    from stable_baselines3.dqn import DQN, MlpPolicy
    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=args.alpha,
        buffer_size=args.buffer,
        exploration_fraction=args.exploration_f,
        exploration_final_eps=args.exploration_f_e,
        batch_size=args.batch_size,
        train_freq = 1,
        verbose=2,
        learning_starts = args.target_net_freq,
        target_update_interval = args.target_net_freq
    )

    # learning of the model
    model.learn(total_timesteps=1000000)
