import gym
import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
sys.path.append("../")
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
import traci

from stable_baselines.common.vec_env import SubprocVecEnv

def make_env(rank, ggi):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SumoEnvironment(net_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection.net.xml',
                              route_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection-vhvh2.rou.xml',
                              out_csv_name='output/2way-single-intersection/ppo2-2way-intersection{}'.format(rank),
                              single_agent=True,
                              use_gui=False,
                              delta_time=20,
                              yellow_time=6,
                              num_seconds=100000,
                              min_green=5,
                              time_to_load_vehicles=120,
                              max_depart_delay=0,
                              ggi=ggi,
                              phases=[
                                  traci.trafficlight.Phase(32000, "GGrrrrGGrrrr"),
                                  traci.trafficlight.Phase(2000, "yyrrrryyrrrr"),
                                  traci.trafficlight.Phase(32000, "rrGrrrrrGrrr"),
                                  traci.trafficlight.Phase(2000, "rryrrrrryrrr"),
                                  traci.trafficlight.Phase(32000, "rrrGGrrrrGGr"),
                                  traci.trafficlight.Phase(2000, "rrryyrrrryyr"),
                                  traci.trafficlight.Phase(32000, "rrrrrGrrrrrG"),
                                  traci.trafficlight.Phase(2000, "rrrrryrrrrry")
                                  ])
        return env
    return _init

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""PPO 2way-Single-Intersection""")
    prs.add_argument("-gam", dest="gamma", type=float, default=0.99, required=False, help="discount factor of PPO.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0005, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.2, required=False, help="clip_range of PPO.\n")
    prs.add_argument("-nst", dest="n_steps", type=int, default=128, required=False, help="n_steps for ppo.\n")
    prs.add_argument("-entc", dest="ent_coef", type=float, default=0.01, required=False, help="clip_range of PPO.\n")
    prs.add_argument("-nmbatch", dest="nminibatches", type=int, default=4, required=False, help="# of mini batches per update.\n")
    prs.add_argument("-vfc", dest="vf_coef", type=float, default=0.5, required=False, help="vf_coefficient of PPO.\n")
    prs.add_argument("-lam", dest="lambdaa", type=float, default=0.95, required=False, help="trade off bias-variance.\n")
    prs.add_argument("-nepoch", dest="noptepochs", type=int, default=4, required=False, help="noptepochs to optimize surrugate.\n")
    prs.add_argument("-w", dest="weight", type=int, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_false", default=True, help="Run GGI algo or not.\n")
    args = prs.parse_args()

    # multiprocess environment
    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(i, ggi) for i in range(n_cpu)])
    reward_space = 4
    weight_coef = args.weight

    if ggi:
        from stable_baselines.ppo2_ggi import PPO2_GGI
        from stable_baselines.common.policies_ggi import MlpPolicy as GGIMlpPolicy
        model = PPO2_GGI(
                policy=GGIMlpPolicy,
                env=env,
                reward_space=reward_space,
                weight_coef=weight_coef,
                gamma=args.gamma,
                n_steps=args.n_steps,
                ent_coef=args.ent_coef,
                verbose=0, 
                learning_rate=args.alpha,
                vf_coef=args.vf_coef,
                lam=args.lambdaa,
                nminibatches=args.nminibatches,
                noptepochs=args.noptepochs,
                cliprange=args.clip_range)
    else:
        from stable_baselines import PPO2
        from stable_baselines.common.policies import MlpPolicy
        model = PPO2(
                policy=MlpPolicy,
                env=env,
                gamma=args.gamma,
                n_steps=args.n_steps,
                ent_coef=args.ent_coef,
                verbose=0, 
                learning_rate=args.alpha, 
                vf_coef=args.vf_coef,
                lam=args.lambdaa,
                nminibatches=args.nminibatches,
                noptepochs=args.noptepochs,
                cliprange=args.clip_range)

    model.learn(total_timesteps=1000000)
