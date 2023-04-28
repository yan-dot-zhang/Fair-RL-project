
# Learning Fair Policies in Multiobjective (Deep) Reinforcement Learning with Average and Discounted Rewards

## Abstract:
As the operations of autonomous systems generally affect  simultaneously several users, it is crucial that their designs account for fairness considerations. In contrast to standard (deep) reinforcement learning (RL), we investigate the problem of learning a policy that treats its users equitably. In this paper, we formulate this novel RL problem, in which an objective function, which encodes a notion of fairness that we formally define, is optimized. For this problem, we provide a theoretical discussion where we examine the case of discounted rewards and that of average rewards. During this analysis, we notably derive a new result in the standard RL setting, which is of independent interest: it states a novel bound on the approximation error with respect to the optimal average reward of that of a policy optimal for the discounted reward.
Since learning with discounted rewards is generally easier, this discussion further justifies finding a fair policy for the average reward by learning a fair policy for the discounted reward. Thus, we describe how several classic deep RL algorithms can be adapted to our fair optimization problem, and we validate our approach with extensive experiments in three different domains.

## Requirements

-   Python >=3.7
-   PyTorch version 1.11
-   OpenAI Gym version: 0.26.1
-   Matplotlib version 3.1.1

## SUMO installation
```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 

```

## DC ENV

For DC environment please check the main [iroko](https://github.com/dcgym/iroko) repository.
## Example
### Run PPO on SC domain:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python predator_prey/ppo_so_aba.py
``` 


### Run DQN on TL domain:
```
python sumo-rl/experiments/dqn_basic.py 
``` 

## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{SiddiqueWengZimmer20,
 author = {Siddique, Umer and Weng, Paul and Zimmer, Matthieu},
 booktitle = {ICML},
 title = {Learning Fair Policies in Multi-Objective (Deep) Reinforcement Learning with Average and Discounted Rewards},
 year = {2020}
}
``` 

