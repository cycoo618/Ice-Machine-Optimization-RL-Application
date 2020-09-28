# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:05:17 2020

@author: ychen
"""


import json


class State:

    def __init__(self, current_hour, newIce, ice_needed, ice_prepared, ice_produced, ice_melted, diff,switch):
        self.current_hour = current_hour
        self.newIce = newIce
        self.ice_needed = ice_needed
        self.ice_prepared = ice_prepared
        self.ice_produced = ice_produced
        self.ice_melted = ice_melted
        self.diff = diff
        self.switch = switch

    def __str__(self):
        return json.dumps(self.__dict__)
