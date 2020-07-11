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
                              route_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection-vhvh1.rou.xml',
                              out_csv_name='output/2way-single-intersection/a2c-2way-intersection{}'.format(rank),
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
                                  description="""A2C 2way-Single-Intersection""")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0005, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-st", dest="steps", type=int, default=5, required=False, help="n steps for A2C.\n")
    prs.add_argument("-entc", dest="ent_coef", type=float, default=0.01, required=False, help="entropy coeff.\n")
    prs.add_argument("-vfc", dest="vf_coef", type=float, default=0.25, required=False, help="vf_coefficient.\n")
    prs.add_argument("-w", dest="weight", type=int, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_false", default=False, help="Run GGI algo or not.\n")
    args = prs.parse_args()
    # multiprocess environment
    n_cpu = 10
    ggi = args.ggi
    env = SubprocVecEnv([make_env(i, ggi) for i in range(n_cpu)])
    reward_space = 4
    weight_coef = args.weight
    if ggi:
        from stable_baselines.a2c_ggi import A2C_GGI
        from stable_baselines.common.policies_ggi import MlpPolicy as GGIMlpPolicy
        model = A2C_GGI(GGIMlpPolicy, env, reward_space, weight_coef, verbose=0, learning_rate=args.alpha,
                    n_steps=args.steps, lr_schedule='constant', vf_coef=args.vf_coef, ent_coef=args.ent_coef)
    else:
        from stable_baselines import A2C
        from stable_baselines.common.policies import MlpPolicy
        model = A2C(MlpPolicy, env, verbose=0, learning_rate=args.alpha, n_steps=args.steps, lr_schedule='constant',
                    vf_coef=args.vf_coef, ent_coef=args.ent_coef)

    model.learn(total_timesteps=1000000)
    
