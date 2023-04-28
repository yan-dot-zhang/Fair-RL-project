import subprocess
import os
import sys

# script path
script_path = './predator_prey/'

# run the experiment
# a2c
subprocess.run(['python', script_path+'a2c_so_aba.py', '-ggi','-st', '10'])
subprocess.run(['python', script_path+'a2c_so_aba.py', '-st', '10'])
print('a2c done')

raise Exception('stop here')
# ppo
subprocess.run(['python', script_path+'ppo_so_aba.py' , '-ggi', '-a', '0.00005'])
subprocess.run(['python', script_path+'ppo_so_aba.py', '-a', '0.001'])
print('ppo done')


# dqn
for i in range(10):
    subprocess.run(['python', script_path+'dqn_so_aba.py' , '-id', str(i), '-a', '0.0001', '-batch', '64'])
for i in range(10):
    subprocess.run(['python', script_path+'dqn_so_aba.py' , '-id', str(i), '-ggi', '-a', '0.005', '-batch', '128'])
print('dqn done')

# random
for i in range(10):
    subprocess.run(['python', script_path+'random_so_aba.py' , '-id', str(i)])
print('random done')