"""
Created on Tue Dec 24 13:55:40 2019 by @author: umer

Adapted on Wed Apr 26, 2023.
"""

# temporarily disable the warnings
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from numpy import linalg


class AbaloneParams():
    def __init__(self):
        # surfaces of the 4 kinds of habitats (in m^2)
        self.areas = np.array([64.78, 109.47, 90.10, 31.35]) * 10e5
        self.total_area = np.sum(self.areas)
        self.area_ratios = self.areas / self.total_area
        # initial population
        self.init_density_best = 0.23  # initial density of abalone
        self.init_density_by_age = np.array([0.047, 0.056, 0.040, 0.023, 0.018,
                                             0.007, 0.011, 0.003, 0.00, 0.025])
        self.max_density = 3.34  # max carrying capacity (#abalone per m^2)
        self.density_by_area = np.linspace(0.835, 3.34, 4)  # density of abalone in 4 habitats
        # survival rates
        self.survival_rate = 0.818  # survival rate of adult abalone
        self.sj_h = 0.857  # survival rate of juvenile abalone in high level
        self.sj_m = 0.542  # survival rate of juvenile abalone in medium level
        self.sj_l = 0.227  # survival rate of juvenile abalone in low level
        # growth rate of abalone in 4 habitats
        self.area_growth_rates = np.linspace(1.05, 1.6, 4)
        self.female_ratio = 0.5  # ratio of female abalone
        # age-specific fertility rates
        self.fertility_rates = np.array([0.136, 0.26, 0.38, 0.491, 0.593,
                                         0.683, 0.745, 0.795, 0.835, 1.166])


class SeaOtterParams():
    def __init__(self):
        self.init_population = 100  # initial population of sea otter
        self.growth_rate = 0.191  # growth rate of sea otter
        self.max_capacity_by_num = 4073  # carrying capacity of sea otter by the total number
        self.poach_threshold = 0.01  # threshold of poaching activity
        self.poach_high = 0.23  # high level of poaching activity
        self.poach_med = 0.17  # medium level of poaching activity
        self.max_num_aba_eaten = 18  # maximum number of abalone eaten by sea otter
        self.isAPoachEfficient = 0  # 0 for No and 1 for Yes
        # low level of poaching activity
        self.poach_low = 0.1 if not self.isAPoachEfficient else 0.05
        self.death_rate_min = 0.23  # minimum death rate of sea otter
        self.death_rate_max = 0.42  # maximum death rate of sea otter
        self.legal_culling_rate = 0.6  # authorise culling when sea otter density is above 60% of carrying capacity
        self.oil_spill_freq = 0.1  # oil spill frequency
        self.extinct_threshold = 10  # threshold of extinct population (abundance)


class PredatorPrey(gym.Env):
    def __init__(self, out_csv_name, ggi, iFR, iFRnum, save_mode='append'):
        self.so = SeaOtterParams()
        self.aba = AbaloneParams()

        # define the observation and action space
        self.actions = 5
        self.action_space = spaces.Discrete(self.actions)
        self.reward_space = spaces.Discrete(2)
        # observation space is a vector of 11 elements, 10 for the 10 age groups of abalone and 1 for the sea otter
        self.observation_space = spaces.Box(low=0, high=self.aba.total_area * self.aba.max_density, shape=(11,),
                                            dtype=np.float32)
        self.ggi = ggi  # whether to use a fairness objective

        # variables to be used
        self.poach_rate_factor = 0
        self.feeding_mode = iFR  # 1: hyperbolic, 2: linear
        self.feeding_level = iFRnum  # 0: low, 1: medium, 2: high

        # initial population
        self.num_aba_by_age = []
        self.num_aba_by_age_female = []
        self.num_so = 100

        # record statistics
        self.abundance = 0  # abundance of sea otter (# / max_capacity)
        self.density = 0  # density of abalone (# / m^2)
        self.metrics = []

        # running parameters
        self.seed_init = False
        self.run = 0
        self.episode_length = 5000  # 13.7 year
        # output file name
        self.out_csv_name = out_csv_name
        # choose how to save the results, either append or write
        if save_mode not in ['append', 'write']:
            raise TypeError(f"{save_mode} is an invalid save mode, please choose between 'append' and 'write'")
        self.save_mode = save_mode

    def calculate_survival_rate_matrix(self):
        """Calculate the survival rate matrix of abalone.

        Returns:
            survival_matrix (np.array): survival rate matrix of abalone

        """
        survival_matrix = np.zeros((10, 10))
        # survival rate of juvenile abalone (age 4)
        survival_matrix[0, :] = self.aba.fertility_rates * self.aba.female_ratio * self.aba.sj_h
        # survival rate of adult abalone (age 5-13)
        for i in range(1, 10):
            survival_matrix[i, i - 1] = self.aba.survival_rate
        survival_matrix[9, 9] = self.aba.survival_rate
        return survival_matrix

    def ini_aba_pop(self):
        """ Initialise the population of abalone.

        Returns:
            num_by_age (np.array): initial population of abalone by age
            num_female_by_age (np.array): initial population of female abalone by age

        """
        # age specific density
        num_female_by_age = self.aba.init_density_by_age * self.aba.female_ratio * self.aba.total_area
        best_density_female = self.aba.init_density_best * self.aba.female_ratio
        best_num_female = best_density_female * self.aba.total_area
        # average growth rate of abalone in 4 habitats (used to model the max growth rate)
        max_r = self.aba.area_ratios @ np.transpose(self.aba.area_growth_rates)
        self.survival_matrix = self.calculate_survival_rate_matrix()
        # assume there are some free time steps for abalone to grow before sea otter is introduced
        for _ in range(50):
            num_female = np.sum(num_female_by_age)
            # predicted population growth rate assumed to follow Beverton-Holt function
            pred_rate = max_r * best_num_female / (max_r * num_female - num_female + best_num_female)
            # eigen values of the survival matrix indicate the evolution of the population
            eig_vals, _ = linalg.eig(self.survival_matrix)
            growth_rate = pred_rate / np.max(eig_vals).real
            self.survival_matrix *= growth_rate
            num_female_by_age = np.dot(self.survival_matrix, num_female_by_age)
        # calculate the total number of abalone by age
        num_by_age = num_female_by_age / self.aba.female_ratio
        return num_by_age, num_female_by_age

    def calculate_female_aba_avg_density(self):
        """ Calculate the average carrying capacity of female abalones.

        Returns:
            density_aba_female(`float`): the density of female abalones.

        """
        # total number of abalones
        num_aba = np.sum(self.aba.density_by_area * self.aba.areas)
        # Average density of abalone
        self.density_aba = num_aba / self.aba.total_area
        # Average carrying capacity females
        density_female_aba = self.density_aba * self.aba.female_ratio
        return density_female_aba

    def seed(self, seed=None):
        """ Seed the environment for reproducibility. """
        # TODO: check if this is necessary.
        if not self.seed_init:
            _, self.env_seed = seeding.np_random(seed)
            self.seed_init = True

    def get_linear_func_response_params(self):
        """ Get the parameters for the linear function response of sea otter.

        Returns:
            response_params (np.array): a 4x2 matrix for the linear predation behavior of sea otter.

        """
        response_params = np.array([[self.density_aba, self.so.max_num_aba_eaten / 3],
                                    [self.density_aba, 2 * self.so.max_num_aba_eaten / 3],
                                    [2 * self.density_aba / 3, self.so.max_num_aba_eaten],
                                    [self.density_aba / 3, self.so.max_num_aba_eaten]])
        return response_params

    def derive_hyperbolic_response(self):
        """ This assumes sea otters are specialist predators feeding mainly on abalone.

        Returns:
            hyper_response_params (np.array): a matrix for the hyperbolic predation behavior.

        """
        # foraging success rates
        success_rate_min = 0.13
        success_rate_max = 0.36
        # total time the animal spent foraging adjusted by the success rate
        total_foraging_time = np.array([0.4 * success_rate_min, 0.4 * success_rate_max, 0.4 * 1])
        # total handling time
        total_handling_time = 166 / 3600 / 24
        consumption_rates = []  # consumption rates
        half_saturation_consts = []  # half saturation constants

        # calculate the consumption rate and half saturation constant for each foraging success rate
        for idx in range(len(total_foraging_time)):
            # attack rate is calculated following the formula in the paper
            attack_rate = self.so.max_num_aba_eaten / (self.density_aba * total_foraging_time[idx] - \
                          self.so.max_num_aba_eaten * total_handling_time * self.density_aba)
            consumption_rates.append(total_foraging_time[idx] / total_handling_time)
            half_saturation_consts.append(1 / (attack_rate * total_handling_time))
        hyper_response_params = np.concatenate([consumption_rates, half_saturation_consts])
        return hyper_response_params

    def reset(self):
        """ Reset the environment to the initial state.

        Returns:
            state (np.array): the initial state of the environment.

        """
        self.step_counter = 0
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self.num_so = 0
        self.num_aba_by_age, self.num_aba_by_age_female = self.ini_aba_pop()
        return np.concatenate([self.num_aba_by_age, [self.num_so]])

    def step(self, action):
        """ Take a step in the environment.

        Args:
            action (`int`): the action to take in the environment.

        Returns:
            state (`np.array`): the state of the environment after taking the action.
            reward (`float`): the reward for taking the action.
            done (`bool`): whether the episode is done.
            info (`dict`): additional information about the environment.

        """
        # adjust the poaching rate at each run to better model randomness
        self.poach_rate_factor = self.adjust_poach_rate(action)

        # update the population
        self.update_abalone_population()
        self.update_sea_otter_population()

        # adjust the population after culling
        if action == 3:
            self.num_so = min(self.so.legal_culling_rate * self.so.max_capacity_by_num, self.num_so)
        if action == 4:
            num_culling = (self.num_so - self.so.legal_culling_rate * self.so.max_capacity_by_num) / 2
            self.num_so = min(self.num_so - num_culling, self.num_so)

        # adjust the population after predation
        if self.num_so != 0:
            self.predation_on_func_response()
            if np.sum(self.num_aba_by_age_female) < 0:
                print('We are in debt, predators are starving')

        # adjust the population after poaching
        self.compute_poaching_impact(self.poach_rate_factor)
        if np.sum(self.num_aba_by_age_female) < 0:
            print('We are in debt, predators are starving')

        # update step counter
        self.step_counter += 1
        # get the state, reward, done, and info
        state = self.compute_obs()
        reward = self.compute_reward()
        done = self.step_counter >= self.episode_length
        info = self.compute_step_info()
        self.metrics.append(info)

        return state, reward, done, info

    def update_abalone_population(self):
        """ Update the abalone population based on the survival matrix and the growth rate.

        Returns:
            None

        """
        # calculate the average female growth rate
        density_female_aba = self.calculate_female_aba_avg_density()
        # calculate the carrying capacity
        best_num_female = density_female_aba * self.aba.total_area
        # calculate the maximum growth rate (by averaging the rates across four areas)
        max_r = self.aba.area_ratios @ np.transpose(self.aba.area_growth_rates)
        # calculate the number of females
        num_female = np.sum(self.num_aba_by_age_female)
        # predicted population growth rate assumed to follow Beverton-Holt function
        pred_rate = max_r * best_num_female / (max_r * num_female - num_female + best_num_female)
        # eigen values of the survival matrix indicate the evolution of the population
        eig_vals, _ = linalg.eig(self.survival_matrix)
        growth_rate = pred_rate / np.max(eig_vals).real
        self.survival_matrix = self.survival_matrix * growth_rate

        # update the population
        self.num_aba_by_age_female = np.dot(self.survival_matrix,
                                            self.num_aba_by_age_female)
        self.num_aba_by_age = self.num_aba_by_age_female / self.aba.female_ratio

    def update_sea_otter_population(self):
        """ Sea otter population follows an exponential growth. Oil spill happens randomly and kills a percentage.

        Returns:
            None

        """
        # ideal population according to logistic growth of sea otters
        self.num_so = self.num_so * np.exp(self.so.growth_rate * (1 - self.num_so / self.so.max_capacity_by_num))
        # oil spill happens randomly and kills a percentage of sea otters
        if np.random.rand() < self.so.oil_spill_freq:
            # random death rate
            death_rate_predict = np.random.uniform(self.so.death_rate_min, self.so.death_rate_max)
            self.num_so -= self.num_so * death_rate_predict
            if self.num_so < self.so.extinct_threshold:
                print('Population of sea otter goes extinct bloody oil spill :-[')
                self.num_so = 0  # die out if below threshold

    def predation_on_func_response(self):
        """ Predation on abalone population based on functional response.

        Returns:
            None

        """
        num_aba = np.sum(self.num_aba_by_age)
        # calculate the number of prey to be removed
        removed_aba = 0
        if self.num_so != 0:
            days = 365  # simulate for 365 days
            density_aba = num_aba / self.aba.total_area
            # calculate the number of prey removed
            if self.feeding_mode == 1:
                hyper_response_params = self.derive_hyperbolic_response()
                consumption_rate = hyper_response_params[self.feeding_level]
                half_saturation_const = hyper_response_params[self.feeding_level + 3]
                removed_aba = consumption_rate * density_aba / (half_saturation_const + density_aba) * days * self.num_so
            elif self.feeding_mode == 2:
                linear_response_params = self.get_linear_func_response_params()
                max_density_aba = linear_response_params[self.feeding_level][0]
                max_num_aba_eaten = linear_response_params[self.feeding_level][1]
                if density_aba < max_density_aba:
                    removed_aba = max_num_aba_eaten * density_aba / max_density_aba * days * self.num_so
                else:
                    # TODO: a mistake made here (same as line 343), check the paper
                    removed_aba = max_num_aba_eaten * density_aba / max_density_aba * days * self.num_so
            else:
                print("Invalid functional response")

        # remove according to the number of prey to be removed
        if removed_aba > num_aba:
            print("population crash")
            self.num_aba_by_age = np.zeros(len(self.num_aba_by_age))
        else:
            # distribute uniformly the number of prey to be removed to different age groups
            self.num_aba_by_age -= removed_aba / 10
            # Check if there are any negative values in the array
            if any(self.num_aba_by_age < 0):
                index = 0
                # Continue looping while there are still negative values in the array
                while any(self.num_aba_by_age < 0):
                    # If the current element is negative, redistribute its value to other elements
                    if self.num_aba_by_age[index] < 0:
                        # Compute the index of the age group to add the negative value to
                        index_used_to_adjust = np.mod(index + 1, len(self.num_aba_by_age)) + 1
                        # Add the negative value to the selected age group
                        self.num_aba_by_age[index_used_to_adjust] -= self.num_aba_by_age[index]
                        # Set the current element to zero
                        self.num_aba_by_age[index] = 0
                    index += 1
                    # If we have reached the end of the array, reset the counter variable
                    if index >= len(self.num_aba_by_age):
                        index = 0
        # update female abalone population
        self.num_aba_by_age_female = self.num_aba_by_age * self.aba.female_ratio

    def compute_poaching_impact(self, poach_rate_factor):
        """ Compute the impact of poaching on the abalone population.

        Args:
            poach_rate_factor (float): poaching rate factor [0, 1]

        Returns:
            None

        """
        # compute the total number of abalone
        sum_aba = np.sum(self.num_aba_by_age)
        # random impact level
        random_impact_level = np.random.rand()
        impact_factor = 0
        if poach_rate_factor > 0:  # High and Medium poaching
            if random_impact_level > 0.5:
                impact_factor = (self.so.poach_high - self.so.poach_low) * poach_rate_factor + self.so.poach_low
            else:
                impact_factor = (self.so.poach_med - self.so.poach_low) * poach_rate_factor + self.so.poach_low
        elif poach_rate_factor == 0:  # Low poaching
            if random_impact_level > 0.5:
                impact_factor = self.so.poach_low + 0.01
            else:
                impact_factor = self.so.poach_low - 0.01
        else:
            print("Invalid poaching")

        # if the impact factor is less than the threshold, then the impact is negligible
        if (sum_aba / self.aba.total_area) < self.so.poach_threshold:
            impact_factor = 0.01

        # compute the impact
        self.num_aba_by_age *= (1 - impact_factor)
        self.num_aba_by_age_female *= (1 - impact_factor)

    def compute_obs(self):
        """ Compute the observation for the current step.

        Returns:
            (`np.array`): The observation for the current step.

        """
        # number of sea otters do not exceed its environment capacity
        self.num_so = min(self.so.max_capacity_by_num, self.num_so)
        # number of abalone do not exceed its environment capacity
        if (np.sum(self.num_aba_by_age) / self.aba.total_area) > self.aba.max_density:
            print("Abalone density exceeds from its capacity")
            # TODO: it looks like a bug, as we are assigning a value to a list
            # It never gets executed as the condition seems never true.
            self.num_aba_by_age = self.aba.max_density
        return np.concatenate([self.num_aba_by_age, [self.num_so]])

    def compute_reward(self):
        """ Reward for the current step represented by the sum of normalized current capacity.

        Returns:
            (`float`): The reward for the current step.

        """
        # sea otter abundance is normalized by its capacity (0-1)
        self.abundance = self.num_so / self.so.max_capacity_by_num
        # abalone density is normalized by its capacity (0-1)
        self.density = np.sum(self.num_aba_by_age) / self.aba.total_area
        if self.ggi:
            return np.append(self.abundance, self.density)
        a = np.append(self.abundance, self.density)
        return np.sum(a)

    def compute_step_info(self):
        """ register the step information and return the metrics.

        Returns:
            (`dict`): A dictionary containing the metrics for the current step.

        """
        return {
            'Sea_Otters': self.abundance,
            'Northern_Abalone': self.density,
            'Sum': self.abundance + self.density,
            'TimeLimit.truncated': self.step_counter >= self.episode_length,
        }

    def adjust_poach_rate(self, action):
        """ Maps the action to a poaching rate adjustment factor [0-low, 1-high].

        Args:
            action: The action to be taken.

        Returns:
            The poaching rate adjustment factor.

        """
        poach_adjustment_rate = 1
        # 1: do nothing
        if action == 0:
            poach_adjustment_rate = 1
        # 2: introduce sea otters
        elif action == 1:
            self.num_so = 100
            poach_adjustment_rate = 0
        # 3: reduce harvesting
        elif action == 2:
            poach_adjustment_rate = 0
        # 4: control sea otters (by direct removing them)
        elif action == 3:
            poach_adjustment_rate = 1
        # 5: half enforce half control
        elif action == 4:
            poach_adjustment_rate = 0.5
        else:
            print("Invalid action")
        return poach_adjustment_rate

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def save_csv(self, out_csv_name, run) -> None:
        """ Used to save the metrics in a csv file.

        Args:
            out_csv_name: the name of the csv file
            run: indicate the  when generating multiple files

        Returns:
            None

        """
        if out_csv_name is not None:
            if self.save_mode == 'write':
                df = pd.DataFrame(self.metrics)
                df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)
            elif self.save_mode == 'append':
                if self.run == 1:
                    df = pd.DataFrame(self.metrics)
                    df.to_csv(out_csv_name + '.csv', index=False)
                else:
                    df = pd.DataFrame(self.metrics)
                    df.to_csv(out_csv_name + '.csv', mode='a', header=False, index=False)
            else:
                raise TypeError('Invalid save mode')
