# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:03:36 2020

@author: ychen
"""


import numpy as np
from state import State
import random

Machine_Actions = [0,12] #[OFF, ON]


ice_needed_rate_per_hour = ['0','0','0','0','0','0','0','0','0.07','0.09','0.02','0.01','0.01','0.01','0.06','0.06','0.06'
                            ,'0.07','0.08','0.09','0.1','0.1','0.1','0.07']
ice_melt_rate=0.2
ice_batch_size = 12
max_storage_size = 120
min_storage_size = 0


class env():

    def __init__(self, num_doctors=8, ice_needed_day_mu=50):
        print("Creating Ice Machine environment...")
        #self.episode = 0
        # Ice Needed/Predicted
        self.newIce=0
        self.ice_needed = 0
        # Ice Produced
        self.ice_prepared=0
        self.ice_melted = 0
        self.ice_produced = 0
        # Left Ice
        self.diff = 0
        self.totalReward = 0
        # hour setting
        self.num_hours = 23
        self.current_hour = 0
        self.terminal = False
        self.ice_needed_day_mu = np.random.randint(5,150)

        self.last_action = 0
        self.switch = 0

        # generating a list of weight of ice will be needed per hour based on normal distribution
        self.ice_needed_hour_list_sim = self.generate_ice_needed(self.ice_needed_day_mu)
        print("Creating Hospital environment...END")


    def generate_ice_needed(self, ice_needed_day_mu: int, sigma=1, seed: int = 104):
        # generate a list of 24 hourly patients coming in
        # based on poisson distribution
        np.random.seed(seed)
        #ice_needed_day=max(5,np.random.normal(ice_needed_day_mu,sigma))
        ice_needed_hour_list = [round(float(x) * ice_needed_day_mu, 1) for x in ice_needed_rate_per_hour]

        return ice_needed_hour_list

    def update_ice(self, add_ice: int):
        assert (add_ice in Machine_Actions)

        epRewards = 0
        epIceDeficient = 0


        """
        Restriction:
        Max Storage of Ice is 120
        Min Storage of Ice  is 0
        Otherwise, the action is invalid, the number of doctors at each department will keep unchanged
        """

        #Ice prepared is coming from last cycle
        #Ice produced is for this cycle, where Ice produced = Ice Prepared - ice melted
        #Calculating Ice produced for this cycle after the melting
        #Calculating the Ice prepared based on the ACTION on this cycle, which will be used for next cycle
        if min_storage_size <= (max(0,self.diff) + add_ice) <= max_storage_size:
            self.ice_prepared = round(self.ice_produced + add_ice,1)
            self.ice_produced = round(max(0,self.ice_prepared-ice_melt_rate),1)
            if self.diff >0:
                self.ice_melted += round(ice_melt_rate,1)
        else:
            print("Current hour: {}".format(self.current_hour))
            print("Diff is {}".format(self.diff))
            print("Invalid movement, the model will keep unchanged")




        # from ice_needed_hour_list, retrieve the volumn of ice will be needed in current cycle
        self.newIce = round(self.ice_needed_hour_list_sim[self.current_hour],1)
        ##Ice Needed Adjusted
        self.ice_needed += round(self.newIce + min(0,self.diff),1)
        ##Ice Diff = cum ice produced - adjusted cum ice needed
        self.diff = round(self.ice_produced-self.ice_needed,1)

        #Count 1 for Ice Deficient when diff is negative - will give negative reward accordingly
        if self.diff < 0:
            epIceDeficient += abs(self.diff)

        if self.last_action ==0 and add_ice == 12:
            self.switch = 1
        else:
            self.switch = 0

        #Minimize ice melted
        #Minimize ice left
        #Penalize ice deficient to make sure there is enough ice produced to cover ice needed
        epRewards += (self.ice_melted * -1)
        epRewards += (self.diff * -1)
        epRewards += (self.switch * -50)
        epRewards += epIceDeficient * -30
        self.totalReward += epRewards

        # changing the current hour
        self.current_hour += 1
        # determine the terminal state
        if self.current_hour == self.num_hours+1:
            self.terminal = True

        self.last_action = add_ice

        return State(self.current_hour-1, self.newIce, self.ice_needed, self.ice_prepared, self.ice_produced
                     , self.ice_melted, self.diff, self.switch), self.terminal, epRewards, epIceDeficient


    def getTotalRewards(self):
        return self.totalReward

    def getState(self):
        return State(self.current_hour,self.newIce, self.ice_needed, self.ice_prepared, self.ice_produced
                     , self.ice_melted, self.diff, self.switch)

    def reset(self, seed: int = 104):
        print("Reset Environment ...")

        self.episode = 0
        # Ice Needed/Predicted
        self.newIce=0
        self.ice_needed = 0
        # Ice Produced
        self.ice_prepared=0
        self.ice_melted = 0
        self.ice_produced = 0
        # Left Ice
        self.diff = 0
        self.totalReward = 0

        np.random.seed(seed)
        self.ice_needed_day_mu = np.random.randint(5,150)
        self.ice_needed_hour_list_sim = self.generate_ice_needed(self.ice_needed_day_mu,seed)
        # hour setting
        self.num_hours = 23
        self.current_hour = 0
        self.terminal = False


        print("Reset Environment ...END")
        #print("Reset: current hour is {}".format(self.current_hour))
        return State(self.current_hour,self.newIce, self.ice_needed, self.ice_prepared, self.ice_produced
                     , self.ice_melted, self.diff, self.switch)
