"""
Created on Tue Dec 24 13:55:40 2019 by @author: umer

Adapted on Wed Apr 26, 2023.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from numpy import linalg


class AbaloneParams():
    def __init__(self):
        self.max_density = 3.34  # max carrying capacity (#abalone per m^2)
        self.density_range = np.linspace(0.835, 3.34, 4)  # density of abalone in 4 habitats
        self.survival_rate = 0.818  # survival rate of abalone
        self.sj_h = 0.0857  # survival rate of juvenile abalone in high level
        self.sj_m = 0.0542  # survival rate of juvenile abalone in medium level
        self.sj_l = 0.0227  # survival rate of juvenile abalone in low level
        self.growth_rate_range = np.linspace(1.05, 1.6, 4)  # growth rate of abalone in 4 habitats
        self.female_ratio = 0.5 # ratio of female abalone
        self.fertility_rates = np.array([0.136, 0.26, 0.38, 0.491, 0.593,
                                0.683, 0.745, 0.795, 0.835, 1.166])  # age-specific fertility rate
        self.areas = np.array([64.78, 109.47, 90.10, 31.35]) * 10e5  # surfaces of the 4 kinds of habitat
        self.total_area = np.sum(self.areas)
        self.area_ratios = self.areas / self.total_area
        self.init_density = 0.23  # initial density of abalone


class SeaOtterParams():
    def __init__(self):
        self.init_population = 100 # initial population of sea otter
        self.growth_rate = 0.191  # growth rate of sea otter
        self.max_capacity_by_num = 4073  # carrying capacity of sea otter by the total number
        self.poach_threshold = 0.01  # threshold of poaching activity
        self.poach_high = 0.23  # high level of poaching activity
        self.poach_med = 0.17  # medium level of poaching activity
        self.isAPoachEfficient = 0  # 0 for No and 1 for Yes
        # low level of poaching activity
        self.poach_low = 0.1 if not self.isAPoachEfficient else 0.05
        # living_area = 1036 * 10e5, # activity area of sea otter
        self.death_rate_min = 0.23  # minimum death rate of sea otter
        self.death_rate_max = 0.42  # maximum death rate of sea otter
        self.legal_culling_rate = 0.6  # authorise culling when sea otter density is above 60% of carrying capacity
        self.oil_spill_freq = 0.1  # oil spill frequency
        self.extinct_threshold = 10  # threshold of extinct population (abundance)

class PredatorPrey(gym.Env):
    def __init__(self, out_csv_name, ggi, iFR, iFRnum, save_mode='append'):
        self.so = SeaOtterParams()
        self.aba = AbaloneParams()

        self.actions = 5

        self.seed_init = False
        self.rand_k = self.aba.areas / self.aba.total_area

        # define the obsevation and action space
        self.action_space = spaces.Discrete(self.actions)
        self.reward_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=self.aba.total_area * self.aba.max_density, shape=(11,),
                                            dtype=np.float32)
        self.episode_lenght = 5000  # 13.7 year

        self.authorise_culling = 0.6  # Authorise culling when SO has reached 0.6*k_so
        self.Pemax = 18
        self.metrics = []
        self.run = 0
        self.out_csv_name = out_csv_name
        self.ggi = ggi
        self.poach_rate_factor = 0
        self.iFR = iFR
        self.iFRnum = iFRnum
        # choose how to save the results, either append or write
        # check save_mode input
        if save_mode not in ['append', 'write']:
            raise TypeError(f"{save_mode} is an invalid save mode, please choose between 'append' and 'write'")
        self.save_mode = save_mode

        self.abundance = 0
        self.density = 0
        self.num_aba_list = []
        self.num_so = 100

    def derive_kabah(self):
        """ this func will return the ~[0.837,1.67,2.5,3.34] which are the
        defaults of abalone carrying capacity"""
        c = self.aba.max_density / 4
        return np.array([c, 2 * c, 3 * c, 4 * c])

    def female_aba_pop(self):
        # age specific eggs per produced by a female abalone
        finit = np.array([0.136, 0.26, 0.38, 0.491, 0.593, 0.683, 0.745, 0.795, 0.835, 1.166])
        finit = finit * self.aba.female_ratio * self.aba.sj_h * 10e5  # millions
        return finit

    def calculate_survival_rate_matrix(self):
        """Calculate the survival rate matrix of abalone.

        Returns:
            survival_matrix (np.array): survival rate matrix of abalone

        """
        survival_matrix = np.zeros((10, 10))
        # survival rate of juvenile abalone (age 4)
        survival_matrix[0, :] = self.aba.fertility_rates * self.aba.fertility_rates * self.aba.sj_h
        # survival rate of adult abalone (age 5-13)
        for i in range(1, 10):
            survival_matrix[i, i - 1] = self.aba.survival_rate
        survival_matrix[9, 9] = self.aba.survival_rate
        return survival_matrix

    def ini_aba_pop(self):
        """ only females initial population """
        # age specific density
        INIT_N = np.array(
            [0.047, 0.056, 0.040, 0.023, 0.018, 0.007, 0.011, 0.003, 0.00, 0.025]) * self.aba.female_ratio * self.aba.total_area
        k_target = self.aba.init_density * self.aba.female_ratio
        k = k_target * self.aba.total_area
        x = self.aba.growth_rate_range
        rma = self.rand_k @ np.transpose(x)
        self.survival_matrix = self.calculate_survival_rate_matrix()
        for _ in range(50):
            all1 = np.sum(INIT_N)
            r1 = rma * k / (rma * all1 - all1 + k)
            v, w = linalg.eig(self.survival_matrix)
            yy = np.append(v, w)
            z = np.max(yy)
            m1 = r1 / z.real
            self.survival_matrix = self.survival_matrix * m1
            INIT_N = np.dot(self.survival_matrix, INIT_N)
        N1f = INIT_N
        N1 = N1f / self.aba.female_ratio
        return N1, N1f

    def ini_so_pop(self):
        otters = 100
        return otters

    def calculate_female_aba_avg_density(self):
        """ Calculate the average carrying capacity of female abalones.
        Returns:
            density_aba_female(`float`): the density of female abalones.

        """
        # total number of abalones
        num_aba = np.sum(self.aba.density_range * self.aba.areas)
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
        v = np.array([[self.density_aba, self.Pemax / 3],
                      [self.density_aba, 2 * self.Pemax / 3],
                      [2 * self.density_aba / 3, self.Pemax],
                      [self.density_aba / 3, self.Pemax]])
        return v

    def derive_hyp_FR(self):
        effmin = 0.13
        effmax = 0.36
        Tp = np.array([0.4 * effmin, 0.4 * effmax, 0.4 * 1])
        Th = 166 / 3600 / 24
        Nmax = self.density_aba
        #        N = [a for a in np.arange(0,1.9186,0.01)]
        #        Z = np.zeros((len(Tp),len(N)))
        #        Y = np.zeros(len(N))
        c_hyp = []
        d_hyp = []

        for k in range(len(Tp)):
            a = self.Pemax / (Nmax * Tp[k] - self.Pemax * Th * Nmax)
            c = Tp[k] / Th
            d = 1 / (a * Th)
            c_hyp.append(c)
            d_hyp.append(d)
        #            for i in range(len(N)):
        #                Z[k,i]= a*N[i]*Tp[k]/(1+a*Th*N[i])
        #                Y[i]=a*N[i]*Tp[k]/(1+a*Th*N[i])
        cmin = c_hyp[0]
        dmin = d_hyp[0]
        cmax = c_hyp[1]
        dmax = d_hyp[1]

        return np.array([cmin, cmax, dmin, dmax])

    def reset(self):
        self.step_counter = 0
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self.num_so = 100
        self.num_aba_list, self.AbaPopF = self.ini_aba_pop()
        return np.append(self.num_aba_list, self.num_so)

    def step(self, action):
        self.step_counter += 1
        # act
        self.poach_rate_factor = self.adjust_poach_rate(action)

        # update abalone
        AbaPopF = self.AbaPopF
        AbaPop = self.num_aba_list
        AbaPop, AbaPopF = self.update_abalone_population(AbaPopF)

        # update SO
        self.num_so = self.update_sea_otter_population()
        if action == 3:
            self.num_so = min(self.authorise_culling * self.so.max_capacity_by_num, self.num_so)
        if action == 4:
            remove = (self.num_so - self.authorise_culling * self.so.max_capacity_by_num) / 2
            self.num_so = min(self.num_so - remove, self.num_so)

        # predation
        if self.num_so != 0:
            #            print("desity before predation", np.sum(AbaPop)/self.aba.total_area)
            AbaPop = self.predation_FR(self.num_so, AbaPop)
            AbaPopF = AbaPop * self.aba.female_ratio
            if np.sum(AbaPopF) < 0:
                print('We are in debt, predators are starving')

        # poaching
        AbaPop, AbaPopF = self.compute_poaching_impact(AbaPop, AbaPopF, self.poach_rate_factor)
        if np.sum(AbaPopF) < 0:
            print('We are in debt, predators are starving')

        self.num_aba_list = AbaPop
        self.AbaPopF = AbaPopF
        state = self.compute_obs()
        reward = self.compute_reward()

        done = self.step_counter >= self.episode_lenght
        info = self.compute_step_info()
        self.metrics.append(info)

        return state, reward, done, info

    def update_abalone_population(self, AbaPopF):
        density_female_aba = self.calculate_female_aba_avg_density()  # scalar value
        r = self.aba.growth_rate_range
        k = density_female_aba * self.aba.total_area  # scalar

        rmax = self.rand_k @ np.transpose(r)
        # uncomment if want a stochastic growth rate
        #        x = np.random.rand()
        #        if x <= self.rand_k[0]:  # One way of simulating stochasticity on *r*
        #            rmax=r[0]
        #        elif x <= (self.rand_k[0] + self.rand_k[1]):
        #            rmax=r[1]
        #        elif x <= (self.rand_k[2]+self.rand_k[1]+self.rand_k[0]):
        #            rmax=r[2]
        #        else:
        #            rmax=r[3]
        all1 = np.sum(AbaPopF)
        r1 = rmax * k / (rmax * all1 - all1 + k)
        v, w = linalg.eig(self.survival_matrix)
        yy = np.append(v, w)
        max_eig_G = np.max(yy)
        m1 = r1 / max_eig_G.real
        self.survival_matrix = self.survival_matrix * m1
        AbaPopF = np.dot(self.survival_matrix, AbaPopF)
        AbaPop = AbaPopF / self.aba.female_ratio

        return AbaPop, AbaPopF

    def update_sea_otter_population(self):
        """ Sea otter population follows an exponential growth. Oil spill happens randomly and kills a percentage.

        Returns:
            new_num_so (float): new number of sea otters

        """
        # ideal population according to logistic growth of sea otters
        new_num_so = self.num_so * np.exp(self.so.growth_rate * (1 - self.num_so / self.so.max_capacity_by_num))
        # oil spill happens randomly and kills a percentage of sea otters
        if np.random.rand() < self.so.oil_spill_freq:
            # random death rate
            death_rate_predict = np.random.uniform(self.so.death_rate_min, self.so.death_rate_max)
            new_num_so -= new_num_so * death_rate_predict
            if new_num_so < self.so.extinct_threshold:
                print('Population of sea otter goes extinct bloody oil spill :-[')
                new_num_so = 0  # die out if below threshold
        return new_num_so

    def predation_FR(self, abundance_predator, Tabundance_prey):
        if abundance_predator == 0:
            new_Tabundance_prey = Tabundance_prey
        else:
            days = 365  # 1 year = 365 days
            sum_Tabundance_prey = np.sum(Tabundance_prey)
            Nd = sum_Tabundance_prey / self.aba.total_area  # Nd = density
            if self.iFR == 1:
                v = self.derive_hyp_FR()
                c = v[self.iFRnum]
                d = v[self.iFRnum + 2]
                removed_prey = c * Nd / (d + Nd) * days * abundance_predator
            elif self.iFR == 2:
                v = self.get_linear_func_response_params()
                d_max = v[self.iFRnum][0]
                Pemax = v[self.iFRnum][1]
                if Nd < d_max:
                    removed_prey = Pemax * Nd / d_max * days * abundance_predator
                else:
                    removed_prey = Pemax * Nd / d_max * days * abundance_predator
            else:
                print("Invalid functional response")

        if removed_prey > sum_Tabundance_prey:
            print("population crash")
            new_Tabundance_prey = np.zeros(len(Tabundance_prey))
        else:
            removed = np.ones(len(Tabundance_prey)) * removed_prey / 10
            new_Tabundance_prey = Tabundance_prey - removed
            if (sum(new_Tabundance_prey < 0) > 0):
                i = 1
                while (sum(new_Tabundance_prey < 0) > 0):
                    if new_Tabundance_prey[i] < 0:
                        index = np.mod(i + 1, len(new_Tabundance_prey)) + 1
                        new_Tabundance_prey[index] = new_Tabundance_prey[index] - new_Tabundance_prey[i]
                        new_Tabundance_prey[i] = 0
                    i += 1
                    if i > len(new_Tabundance_prey):
                        i = 1
        return new_Tabundance_prey

    def compute_poaching_impact(self, num_aba_by_age, num_aba_female_by_age, poach_rate_factor):
        """ Compute the impact of poaching on the abalone population.

        Args:
            num_aba_by_age (list): number of abalone by age (4-13)
            num_aba_female_by_age (list): number of female abalone by age (4-13)
            poach_rate_factor (float): poaching rate factor [0, 1]

        Returns:
            num_aba_by_age (list): adjusted number of abalone by age (4-13)
            num_aba_female_by_age (list): adjusted number of female abalone by age (4-13)
        """
        # compute the total number of abalone
        sum_aba = np.sum(num_aba_by_age)
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
        num_aba_by_age *= (1 - impact_factor)
        num_aba_female_by_age *= (1 - impact_factor)
        return num_aba_by_age, num_aba_female_by_age

    def compute_obs(self):
        """ Compute the observation for the current step.

        Returns:

        """
        # number of sea otters do not exceed its environment capacity
        self.num_so = min(self.so.max_capacity_by_num, self.num_so)
        # number of abalone do not exceed its environment capacity
        if (np.sum(self.num_aba_list) / self.aba.total_area) > self.aba.max_density:
            print("Abalone density exceeds from its capacity")
            # TODO: it looks like a bug, as we are assigning a value to a list
            # It never gets executed as the condition seems never true.
            self.num_aba_list = self.aba.max_density
        return np.concatenate([self.num_aba_list, [self.num_so]])

    def compute_reward(self):
        """ Reward for the current step represented by the sum of normalized current capacity.

        Returns:
            (`float`): The reward for the current step.

        """
        # sea otter abundance is normalized by its capacity (0-1)
        self.abundance = self.num_so / self.so.max_capacity_by_num
        # abalone density is normalized by its capacity (0-1)
        self.density = np.sum(self.num_aba_list) / self.aba.total_area
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
            'Sum': self.abundance + self.density
        }

    def adjust_poach_rate(self, action):
        """ Maps the action to a poaching rate adjustment factor [0-low, 1-high].

        Args:
            action: The action to be taken.

        Returns:
            The poaching rate adjustment factor.

        """
        poach_adjustment_rate = 1
        # do nothing
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

    def _set_poach(self, poach):
        self.poach_rate_factor = poach

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
