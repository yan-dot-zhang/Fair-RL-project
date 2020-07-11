import argparse
import os
import sys


from so_abalone_env import PredatorPrey 

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description=""" Random """)
    prs.add_argument("-fr", dest="ifr", type=int, default=1, required=False, help="Functional Response for SC\n")
    prs.add_argument("-fnum", dest="ifrnum", type=int, default=2, required=False, help="Functional Response Num for SC\n")

    args = prs.parse_args()

    ggi = False
    env = PredatorPrey(out_csv_name='results/reward_random', ggi=ggi, iFR=args.ifr, iFRnum=args.ifrnum)
    obs = env.reset()
    for i in range(100000):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            env.reset()


