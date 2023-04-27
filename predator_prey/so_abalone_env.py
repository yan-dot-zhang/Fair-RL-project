"""
Created on Tue Dec 24 13:55:40 2019

@author: umer
"""
import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding
from numpy import linalg as LA


class PredatorPrey(gym.Env):
    def __init__(self, out_csv_name, ggi, iFR, iFRnum, save_mode = 'append'):    
        self.actions = 5
        self.isAPoachEfficient = 0                              # 0 for No and 1 for Yes
        # intrinsic growths
        self.sogrowth = 0.191                                   # groth rate of SO
        self.abagrowth = 1.6                                    # Max growth rate of Abalone
        # carrying capacity
        self.ocapacoty = 4073                                   #number to density
        self.acapacity = 3.34                                   #density
        self.ratio_mf=0.5

        self.avg_abadensity = 0.21
        self.k_init=0.23
        
        self.seed_init = False
         #do not change  current estimate of poaching activity
        self.poach_thr=0.01                                # Poaching threshold e.g. density of abalone is very low 
        self.poach_high=0.23                                    #Poaching intensity 0.20+-0.3 
        self.poach_med=0.17                                   #Poaching intensity 0.20+-0.3
        self.proba_spills = 0.1                                 # To initialize aba population
        
        self.survival_rate = 0.818                              #abalone survival rate
        # survival of juvenile abalone,
        self.sj_h = 8.5700e-07
        self.sj_m = 5.4200e-07
        self.sj_l = 2.2700e-07
               
        # do not change/ specific to pacific rim national park, BC, Canada.
        self.surf_k=np.array([64.78, 109.47, 90.10, 31.35])*10e5  # surfaces of the 4 kinds of habitat
        self.area_aba=np.sum(self.surf_k)                                # total area of abalone
        self.aba_capa = (self.area_aba * self.acapacity)/2
        self.rand_k=self.surf_k/self.area_aba
        
        # define the obsevation and action space
        self.action_space = spaces.Discrete(self.actions)
        self.reward_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=self.area_aba*self.acapacity, shape=(11,),
                                            dtype=np.float32)
        self.episode_lenght = 5000 # 13.7 year

        self.area_so = 1036*10e5
        self.authorise_culling=0.6                                       #Authorise culling when SO has reached 0.6*k_so
        self.oil_spill_frequency=0.1
        self.dead_prct_min=0.23
        self.dead_prct_max=0.42
        self.extinct_when=10                                              # threshold of extinct population (abundance)
        self.Pemax=18
        self.metrics = []
        self.run = 0
        self.out_csv_name = out_csv_name
        self.ggi = ggi
        self.p = 0
        self.iFR = iFR
        self.iFRnum = iFRnum
        # choose how to save the results, either append or write
        # check save_mode input
        if save_mode not in ['append', 'write']:
            raise TypeError(f"{save_mode} is an invalid save mode, please choose between 'append' and 'write'")
        self.save_mode = save_mode
        
    def derive_rmax(self):
        """ this func will return the ~[1.05,1.2,1.4,1.6] which are the
        defaults of max abalone growths"""
        if self.abagrowth < 1.05:
            print("Invalid value of abalone growth rate")
            exit()
        b= 1.05;
        a = (self.abagrowth - b)/3;
        return np.array([b, a+b, 2*a+b, self.abagrowth])
    
    def derive_kabah(self):
        """ this func will return the ~[0.837,1.67,2.5,3.34] which are the
        defaults of abalone carrying capacity"""
        c = self.acapacity/4
        return np.array([c, 2*c, 3*c, 4*c])
    
    def female_aba_pop(self):
        # age specific eggs per produced by a female abalone
        finit = np.array([0.136,0.26,0.38,0.491,0.593,0.683,0.745,0.795,0.835,1.166]) 
        finit = finit * self.ratio_mf * self.sj_h *  10e5 # millions
        return finit

    def Ginith(self):
        G = np.zeros((10,10))
        G[0,:] = self.female_aba_pop()
        G[1,0] = self.survival_rate
        G[2,1] = self.survival_rate
        G[3,2] = self.survival_rate
        G[4,3] = self.survival_rate
        G[5,4] = self.survival_rate
        G[6,5] = self.survival_rate
        G[7,6] = self.survival_rate
        G[8,7] = self.survival_rate
        G[9,8] = self.survival_rate
        G[9,9] = self.survival_rate
        return G
        
    def ini_aba_pop(self):
        """ only females initial population """
        # age specific density
        INIT_N=np.array([0.047, 0.056, 0.040, 0.023,  0.018, 0.007,0.011, 0.003, 0.00, 0.025])*self.ratio_mf*self.area_aba
        k_target=self.k_init*self.ratio_mf
        k=k_target*self.area_aba
        x = self.derive_rmax()
        rma=self.rand_k @ np.transpose(x)
        self.G = self.Ginith()
        for _ in range(50):
            all1 = np.sum(INIT_N)
            r1=rma*k/(rma*all1-all1+k)
            v,w = LA.eig(self.G)
            yy = np.append(v,w)
            z = np.max(yy)
            m1= r1/z.real
            self.G = self.G * m1
            INIT_N=np.dot(self.G,INIT_N)
        N1f = INIT_N
        N1 = N1f/self.ratio_mf
        return N1, N1f
    
    def ini_so_pop(self):
        otters = 100
        return otters
    
    def aba_avg_carry_capacity(self):
        """ compute Average carrying capacity for females abalones """
        k_aba_h = self.derive_kabah()
        d = np.sum(k_aba_h*self.surf_k)
        self.k_aba = d/self.area_aba
        k_aba_fem= self.k_aba*self.ratio_mf   #Average carrying capacity females
        return k_aba_fem
    
    def seed(self, seed=None):
        if not self.seed_init:
            self.np_random, seed = seeding.np_random(seed)
            self.seed_init = True
            self.env_seed = seed
        return self.env_seed
    
    def PARAM_LINEAR_FR(self):
        v = np.array([[self.k_aba, self.Pemax/3],
                      [self.k_aba, 2*self.Pemax/3],
                      [2*self.k_aba/3, self.Pemax],
                      [self.k_aba/3, self.Pemax]])
        return v   
    
    def derive_hyp_FR(self):
        effmin=0.13
        effmax=0.36
        Tp= np.array([0.4*effmin, 0.4*effmax, 0.4*1])
        Th=166/3600/24
        Nmax = self.k_aba
#        N = [a for a in np.arange(0,1.9186,0.01)]
#        Z = np.zeros((len(Tp),len(N)))
#        Y = np.zeros(len(N))
        c_hyp=[]
        d_hyp=[]
        
        for k in range(len(Tp)):
            a= self.Pemax/(Nmax* Tp[k]-self.Pemax*Th*Nmax)
            c=Tp[k]/Th
            d=1/(a*Th)
            c_hyp.append(c)
            d_hyp.append(d)
#            for i in range(len(N)):
#                Z[k,i]= a*N[i]*Tp[k]/(1+a*Th*N[i])
#                Y[i]=a*N[i]*Tp[k]/(1+a*Th*N[i])
        cmin=c_hyp[0]
        dmin=d_hyp[0]
        cmax=c_hyp[1]
        dmax=d_hyp[1]
        
        return np.array([cmin,cmax,dmin,dmax])
            
    def reset(self):     
        self.step_index = 0
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []
        
        if self.isAPoachEfficient ==0:
            self.poach_low=0.1
        else:
            self.poach_low=0.05
        if self.ocapacoty < 500:
            print("Too less Sea otters,  program not designed for small populations")
        self.abalone, self.AbaPopF = self.ini_aba_pop() 
        self.otters = 0
        return np.append(self.abalone, self.otters)        

    def step(self, action):
        self.step_index +=1
        #act
        self.p = self._take_action(action)

        #update abalone
        AbaPopF = self.AbaPopF 
        AbaPop = self.abalone
        AbaPop, AbaPopF = self.northern_abalone_growth_t(AbaPopF)
        
        #update SO
        self.otters, oil = self.sea_otter_growth(self.otters)
        if action == 3:
            self.otters = min(self.authorise_culling * self.ocapacoty ,self.otters)
        if action == 4:
            remove=(self.otters - self.authorise_culling * self.ocapacoty)/2
            self.otters = min(self.otters - remove, self.otters)

        # predation
        if self.otters != 0:
#            print("desity before predation", np.sum(AbaPop)/self.area_aba)
            AbaPop = self.predation_FR(self.otters, AbaPop)
            AbaPopF = AbaPop * self.ratio_mf
            if np.sum(AbaPopF) < 0:
                print('We are in debt, predators are starving')

        #poaching
        AbaPop, AbaPopF = self.compute_poaching_impact(AbaPop, AbaPopF,self.p)
        if np.sum(AbaPopF) < 0:            
            print('We are in debt, predators are starving')

        self.abalone = AbaPop
        self.AbaPopF = AbaPopF
        state = self.compute_obs()
        reward = self.compute_reward()
        
        done = self.step_index >= self.episode_lenght
        info = self._compute_step_info()
        self.metrics.append(info)
        
        return state, reward, done, info
    
    def northern_abalone_growth_t(self, AbaPopF):
        k_aba_f = self.aba_avg_carry_capacity() # scalar value
        r = self.derive_rmax() # vector(4)~ growths
        k = k_aba_f*self.area_aba # scalar
        
        rmax =self.rand_k @ np.transpose(r)
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
        all1=np.sum(AbaPopF)
        r1=rmax*k/(rmax*all1-all1+k)
        v,w = LA.eig(self.G)
        yy = np.append(v,w)
        max_eig_G = np.max(yy)
        m1= r1/max_eig_G.real
        self.G = self.G * m1
        AbaPopF = np.dot(self.G,AbaPopF)
        AbaPop = AbaPopF/self.ratio_mf

        return AbaPop, AbaPopF
    
    def sea_otter_growth(self, N):
        OS = 0
        Y = N*np.exp(self.sogrowth*(1-N/self.ocapacoty))
        oil_spill = np.random.rand()
        if oil_spill < self.oil_spill_frequency:
            dead_prct = self.dead_prct_min + (self.dead_prct_max-self.dead_prct_min)*np.random.rand()
            OS = dead_prct
            Y = Y - Y *dead_prct
            if Y < self.extinct_when:
                print('Population of sea otter goes extinct bloody oil spill :-[')
                Y=0
        return Y, OS
            
    def predation_FR(self, abundance_predator,Tabundance_prey):
        if abundance_predator == 0:
            new_Tabundance_prey=Tabundance_prey
        else:
            days=365   # 1 year = 365 days
            sum_Tabundance_prey = np.sum(Tabundance_prey)
            Nd = sum_Tabundance_prey/self.area_aba    # Nd = density
            if self.iFR == 1:
                v = self.derive_hyp_FR()
                c = v[self.iFRnum]
                d = v[self.iFRnum+2]
                removed_prey=c*Nd/(d+Nd)*days*abundance_predator
            elif self.iFR == 2:
                v = self.PARAM_LINEAR_FR()
                d_max = v[self.iFRnum][0]
                Pemax = v[self.iFRnum][1]
                if Nd < d_max:
                    removed_prey=Pemax*Nd/d_max*days*abundance_predator
                else:
                    removed_prey=Pemax*Nd/d_max*days*abundance_predator
            else:
                print("Invalid functional response")
            
        if removed_prey>sum_Tabundance_prey:
            print("population crash")
            new_Tabundance_prey = np.zeros(len(Tabundance_prey))
        else:
            removed=np.ones(len(Tabundance_prey))*removed_prey/10
            new_Tabundance_prey=Tabundance_prey-removed
            if (sum(new_Tabundance_prey<0)>0):
                i=1
                while (sum(new_Tabundance_prey<0)>0):
                    if new_Tabundance_prey[i]<0:
                        index=np.mod(i+1,len(new_Tabundance_prey))+1
                        new_Tabundance_prey[index]=new_Tabundance_prey[index]-new_Tabundance_prey[i]
                        new_Tabundance_prey[i]=0
                    i +=1
                    if i > len(new_Tabundance_prey):
                        i = 1
        return new_Tabundance_prey
                    
            
    def compute_poaching_impact(self, N, Nf, poach):
        all1 = np.sum(N)
        impact_poaching = np.random.rand()
        if poach>0:  # High and Medium poaching
            if impact_poaching > 0.5:
                impact=(self.poach_high - self.poach_low)* poach + self.poach_low
            else:
                impact=(self.poach_med-self.poach_low)* poach + self.poach_low
        elif poach==0:  #Low poaching
            if impact_poaching > 0.5:
                impact=self.poach_low + 0.01
            else:
                impact=self.poach_low - 0.01
        else:
            print("Invalid poaching")
        #check thershold
        if (all1/self.area_aba) < self.poach_thr:
            impact=0.01
        N = (1-impact)*N
        Nf = (1-impact)*Nf # females
        
        return N, Nf
                
        
    def compute_obs(self):
        if self.otters > self.ocapacoty:
            self.otters = self.ocapacoty
        if (np.sum(self.abalone)/self.area_aba) > self.acapacity:
            print("Abalone density exceeds from its capacity")
            self.abalone = self.acapacity
        return np.append(self.abalone, self.otters)
    
    def compute_reward(self):
        self.abundance = self.otters/self.ocapacoty
        self.density = np.sum(self.abalone)/self.area_aba
        if self.ggi:
            return np.append(self.abundance, self.density)
        a = np.append(self.abundance, self.density)
        return np.sum(a)
    
    def _compute_step_info(self):
        return {
            'Sea_Otters': self.abundance,
            'NOrthern_Abalone': self.density,
            'Sum': self.abundance+self.density
        }
    
    def _take_action(self, action):
        # do nothing
        if action==0:
            paoch = 1
        # introduce sea otters
        elif action == 1:
            # check the recovery of abalone
            oters = self.ini_so_pop()
            self.otters = oters
            paoch = 0                                    
        # enforce paoching
        elif action==2:
            #reduce harvesting
            paoch = 0
        # control sea otters(by direct removing them)
        elif action==3:
            paoch = 1
        # half enforce half control
        elif action==4:
            paoch = 0.5       
        else:
            print("Invalid action")
        
        return paoch
    
    def _set_poach(self,poach):
        self.p = poach
        
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def save_csv(self, out_csv_name, run):
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


 
