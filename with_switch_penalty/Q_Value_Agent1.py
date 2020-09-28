# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:54:30 2020

@author: ychen
"""



import numpy as np
import random
import pandas as pd
from typing import Dict

import state

from state import State
from ice_env import env
import json

models = ['fixed_policy', 'random_policy', 'max_qValue_policy']


class agent():
    """
       Defines a state-action table that can be used to store Q-values or action counts.
       """

    q_table: Dict[str, Dict[int, float]]  # Q-value table   [Action, q-value]
    state_action_counts: Dict[str, Dict[int, int]]
    #column_names = ['Episode','Current_Hour','New_Ice_Needed','Ice_Needed', 'Ice_Prepared', 'Ice_Melted', 'Ice_Produced', 'Diff',
    #                    'Action', 'Reward', 'Q_val']
    #episode_df = pd.DataFrame(columns=column_names)

    #    state_action_counts: Dict[State, Dict[int, int]]  # The number of updates to each state-action pair

    def __init__(self, alpha=0.1, gamma=1.0, numGames=10):
        print('Created an Agent ...')
        self.actions = [0,12]
        self.q_table = dict()
        self.episode_df = pd.DataFrame(columns=['Episode','Current_Hour','New_Ice_Needed','Ice_Needed', 'Ice_Prepared', 'Ice_Melted'
                                       , 'Ice_Produced', 'Diff','Action', 'Reward', 'Q_val'])
        self.state_action_counts = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.numGames = numGames
        self.totalReward = dict()
        self.totalMelted = dict()
        self.totalDeficient = dict()
        self.totalDiff = dict()
        self.totalNeeded = dict()
        self.totalProduced = dict()
        self.totalWasted = dict()
		#self.totalDischargedtoGenerated = dict()
        print('Created an Agent ...END')

    def getGreedyPolicy(self):
        COLUMN_NAMES = ['Current_Hour','New_Ice_Needed','Ice_Needed', 'Ice_Prepared', 'Ice_Melted', 'Ice_Produced', 'Diff',
                        'Action', 'Q_val']
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for pk, pv in self.q_table.items():
            state = json.loads(pk)
            action = max(pv, key=pv.get)
            df = df.append({'Current_Hour':state.get('current_hour'),
                            'New_Ice_Needed':state.get('newIce'),
                            'Ice_Needed': state.get('ice_needed'),
                            'Ice_Prepared': state.get('ice_prepared'),
                            'Ice_Melted': state.get('ice_melted'),
                            'Ice_Produced': state.get('ice_produced'),
                            'Diff': state.get('diff'),
                            'Action': action,
                            'Q_val': pv.get(action)},
                           ignore_index=True)
        return df

    def getQValueTable(self):
        COLUMN_NAMES = ['Current_Hour','New_Ice_Needed','Ice_Needed', 'Ice_Prepared', 'Ice_Melted', 'Ice_Produced'
                        , 'Diff', 'Switch', 'Action', 'Q_val']
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for pk, pv in self.q_table.items():
            for k, v in pv.items():
                state = json.loads(pk)
                df = df.append({'Current_Hour':state.get('current_hour'),
                                'New_Ice_Needed':state.get('newIce'),
                                'Ice_Needed': state.get('ice_needed'),
                                'Ice_Prepared': state.get('ice_prepared'),
                                'Ice_Melted': state.get('ice_melted'),
                                'Ice_Produced': state.get('ice_produced'),
                                'Diff': state.get('diff'),
                                'Switch': state.get('switch'),
                                'Action': k,
                                'Q_val': v},
                               ignore_index=True)
        return df

    def maxAction(self, state: State, actions, model: str):
        if model.casefold() == 'fixed_policy'.casefold():
            return 0
        elif model.casefold() == 'random_policy'.casefold():
            return random.choice(actions)

        if self.q_table is None:
            return random.choice(actions)
        elif self.q_table.get(state.__str__()) is None:
            return random.choice(actions)
        else:  # found one state corresponding actions and values
            if len(self.q_table[state.__str__()]) == len(actions):
                # get max value actions
                # valuesPairs = np.array(self.q_table[state.__str__()])
                action = max(self.q_table[state.__str__()], key=self.q_table[state.__str__()].get)
                return action
            else:
                newAction = list()
                keys = self.q_table[state.__str__()].keys()
                for action in actions:
                    if action not in keys:
                        newAction.append(action)
                return random.choice(newAction)

    def getValueFromStateAndAction(self, state: State, action, reward: int = 0, isExecutedAction: bool = False):
        if self.q_table.get(state.__str__()) is None:
            q_act_dict = dict()
            q_act_dict[action] = reward
            if isExecutedAction:
                self.q_table[state.__str__()] = q_act_dict
        else:
            if self.q_table[state.__str__()].get(action) is None:
                if isExecutedAction:
                    self.q_table[state.__str__()][action] = reward

        if (self.q_table.get(state.__str__()) is None) or (self.q_table.get(state.__str__()).get(action) is None):
            return 0
        else:
            return self.q_table.get(state.__str__()).get(action)

    def updateStateActionCounts(self, state: State, action: int):
        if self.state_action_counts.get(state.__str__()) is None:
            # First time tried this action for this state, initial dict
            q_count_dict = dict()
            q_count_dict[action] = 1
            self.state_action_counts[state.__str__()] = q_count_dict
        elif self.state_action_counts[state.__str__()].get(action) is None:
            # First time tried this action for this state, update count for this action
            self.state_action_counts[state.__str__()][action] = 1
        else:
            self.state_action_counts[state.__str__()][action] = self.state_action_counts[state.__str__()].get(
                action) + 1

    def getCountsFromStateAndAction(self, state: State, action: int):
        if self.state_action_counts.get(state.__str__()) is None:
            return 1
        elif self.state_action_counts[state.__str__()].get(action) is None:
            return 1
        else:
            return self.state_action_counts[state.__str__()][action]




    def run(self, alpha: float, gamma: float, numGames: int, env: env, model: str = 'max_qValue_policy'):
        assert (model in models)
        seed = 41
        self.alpha = alpha
        self.gamma = gamma
        self.numGames = numGames
        #self.totalReward = np.zeros(numGames)
        for i in range(self.numGames):
            print('Starting game ', i)
            epsReward = 0
            epIceDeficient = 0
            epIceMelted = 0
            epDiff = 0

            terminal = False
            seed += 1
            state = env.reset(seed)
            while not terminal:
                # choosing the max action
                action = self.maxAction(state, self.actions, model)
                state_, terminal, reward, iceDeficient = env.update_ice(action)
                epsReward += reward
                epIceDeficient += iceDeficient
                action_ = self.maxAction(state_, self.actions, model)
                value = self.getValueFromStateAndAction(state, action, isExecutedAction=True) + alpha * (
                        reward + gamma * self.getValueFromStateAndAction(state_, action_) - self.getValueFromStateAndAction(
                    state, action)
                )

                count = self.getCountsFromStateAndAction(state, action)
                updatedValue = value / 2

                # updating Q table
                self.q_table[state.__str__()][action] = updatedValue
                # updating action counts table
                self.updateStateActionCounts(state, action)

                state_c = json.loads(state.__str__())

                self.episode_df = self.episode_df.append({'Episode': i,
                                                          'Current_Hour':state_c.get('current_hour'),
                                                          'New_Ice_Needed':state_c.get('newIce'),
                                                          'Ice_Needed': state_c.get('ice_needed'),
                                                          'Ice_Prepared': state_c.get('ice_prepared'),
                                                          'Ice_Melted': state_c.get('ice_melted'),
                                                          'Ice_Produced': state_c.get('ice_produced'),
                                                          'Diff': state_c.get('diff'),
                                                          'Switch': state_c.get('switch'),
                                                          'Action': action,
                                                          'Reward':reward,
                                                          'Q_val': updatedValue},
                                                         ignore_index=True)

                state = state_

            state_l = json.loads(state.__str__())
            self.episode_df = self.episode_df.append({'Episode': i,
                                                      'Current_Hour':state_l.get('current_hour'),
                                                      'New_Ice_Needed':state_l.get('newIce'),
                                                      'Ice_Needed': state_l.get('ice_needed'),
                                                      'Ice_Prepared': state_l.get('ice_prepared'),
                                                      'Ice_Melted': state_l.get('ice_melted'),
                                                      'Ice_Produced': state_l.get('ice_produced'),
                                                      'Diff': state_l.get('diff'),
                                                      'Switch': state_l.get('switch'),
                                                      'Action': '',
                                                      'Reward':'',
                                                      'Q_val': ''},
                                                     ignore_index=True)

            epIceMelted = state_l.get('ice_melted')
            epDiff = state_l.get('diff')

            self.totalReward[i] = epsReward
            self.totalMelted[i] = epIceMelted
            self.totalDeficient[i] = epIceDeficient
            self.totalDiff[i] = epDiff
            self.totalNeeded[i] = state_l.get('ice_needed')
            self.totalProduced[i] = state_l.get('ice_produced')
            self.totalWasted[i] = self.totalDiff[i]+self.totalMelted[i]


        print('training end')


    def reset(self):
        print('Reset agent ...')
        self.q_table = dict()
        self.state_action_counts = dict()
        #self.totalReward = np.zeros(self.numGames)
        print('Reset agent ...END')
