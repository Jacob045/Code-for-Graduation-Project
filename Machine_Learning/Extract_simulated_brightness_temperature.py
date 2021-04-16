#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/16 19:50
# @File     : Extract_simulated_brightness_temperature.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com

import pandas as pd
import os

def extract():
    Dir_path = r'E:\test\test'
    Files_paths = []
    for root,dirs,files in os.walk(Dir_path):
        for file in files:
            Files_paths.append(os.path.join(root,file))

    Cols = ['22.235', '23.035', '23.835', '26.235', '30.000', '51.250',
           '52.280', '53.850', '54.940', '56.660', '57.290', '58.800']
    data = pd.DataFrame(columns=Cols)

    for Files_path in Files_paths:
        dat_file = pd.read_csv(Files_path, sep='\s+', index_col='Time')
        data = data.append(dat_file)

    return data