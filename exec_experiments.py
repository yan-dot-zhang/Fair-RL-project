import subprocess
import os
import sys

# script path
script_path = './predator_prey/'

# run the experiment
# a2c
subprocess.run(['python', script_path+'a2c_so_aba.py', '-ggi'])
subprocess.run(['python', script_path+'a2c_so_aba.py'])
print('a2c done')

# ppo
subprocess.run(['python', script_path+'ppo_so_aba.py' , '-ggi'])
subprocess.run(['python', script_path+'ppo_so_aba.py'])
print('ppo done')
raise Exception('stop here')

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
