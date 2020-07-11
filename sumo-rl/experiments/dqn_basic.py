import gym
import numpy as np
import os
import sys
import argparse
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
sys.path.append("../")
from gym import spaces
from sumo_rl.environment.env import SumoEnvironment
import traci
import pandas as pd


if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Deep-Q-Learning 2way-Single-Intersection""")
    prs.add_argument("-a", dest="alpha", type=float, default=0.001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-b", dest="buffer", type=int, default=50000, required=False, help="Buffer size of DQN.\n")
    prs.add_argument("-ef", dest="exploration_f", type=float, default=0.1, required=False, help="exploration_fraction.\n")
    prs.add_argument("-efs", dest="exploration_f_e", type=float, default=0.02, required=False, help="exploration_final_eps.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=128, required=False, help="Batch size for NN.\n")
    prs.add_argument("-targfr", dest="target_net_freq", type=int, default=500, required=False, help="Update freq for target.\n")
    prs.add_argument("-w", dest="weight", type=int, default=2, required=False, help="Weight coefficient\n")
    prs.add_argument("-ggi", action="store_false", default=False, help="Run GGI algo or not.\n")
    args = prs.parse_args()
    
    ggi = args.ggi
    env = SumoEnvironment(net_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection.net.xml',
                          route_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection-vhvh.rou.xml', 
                          out_csv_name='output/2way-single-intersection/dqn-2way-intersection',
                          single_agent=True,
                          use_gui=False,
                          num_seconds=100000,
                          time_to_load_vehicles=120,
                          max_depart_delay=0,
                          ggi=ggi,
                          phases=[         #4 possible predetermined phases
                              traci.trafficlight.Phase(32000, "GGrrrrGGrrrr"), # NS green action 0
                              traci.trafficlight.Phase(2000, "yyrrrryyrrrr"),  # NS yellow
                              traci.trafficlight.Phase(32000, "rrGrrrrrGrrr"), # NSL Greeb action 1
                              traci.trafficlight.Phase(2000, "rryrrrrryrrr"),  # NSL yellow
                              traci.trafficlight.Phase(32000, "rrrGGrrrrGGr"), # East west green action 2
                              traci.trafficlight.Phase(2000, "rrryyrrrryyr"),  # East west yellow
                              traci.trafficlight.Phase(32000, "rrrrrGrrrrrG"), # East west Left turn green action 3
                              traci.trafficlight.Phase(2000, "rrrrryrrrrry")   # East west YEllow
                              ])

    if ggi:
        from stable_baselines.deepq_ggi import DQN_GGI
        from stable_baselines.deepq_ggi import MlpPolicy as GGIMlpPolicy
        model = DQN_GGI(
            env=env,
            policy=GGIMlpPolicy,
            weight_coef = args.weight,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            double_q = False,
            target_network_update_freq = args.target_net_freq
        )
    else:
        from stable_baselines.deepq import DQN, MlpPolicy
        model = DQN(
            env=env,
            policy=MlpPolicy,
            learning_rate=args.alpha,
            buffer_size=args.buffer,
            exploration_fraction=args.exploration_f,
            exploration_final_eps=args.exploration_f_e,
            batch_size=args.batch_size,
            double_q = False,
            target_network_update_freq = args.target_net_freq
        )
    # learning of the model
    model.learn(total_timesteps=100000)
    

