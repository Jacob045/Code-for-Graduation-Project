#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/17 15:39
# @File     : Compare_bt.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com

from  Extract_Observation_bt import Extract_observation_bt
from Extract_Simulated_bt import Extract_simulated_bt

Years = ['2007']
Observation_bt = Extract_observation_bt(Years)
Simulated_bt = Extract_simulated_bt()

print(Observation_bt)