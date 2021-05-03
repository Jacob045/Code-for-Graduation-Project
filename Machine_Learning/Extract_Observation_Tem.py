#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/29 19:47
# @File     : Extract_Observation_Tem.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com
import os
import pandas as pd


def extract_observation_tem(years):
    dir_path = r'I:\Data\sn\Yuzhong_1989_2020_MWMOD'
    files_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file[30:34] in years:
                files_paths.append(os.path.join(root, file))

    cols = ['1965', '2065', '2165', '2265', '2365', '2465', '2565', '2665',
            '2765', '2865', '2965', '3215', '3465', '3715', '3965', '4215',
            '4465', '4715', '4965', '5215', '5465', '5715', '5965', '6215',
            '6465', '6715', '6965', '7215', '7465', '7715', '7965', '8215',
            '8465', '8715', '8965', '9215', '9465', '9715', '9965', '10215',
            '10465', '10715', '10965', '11215', '11465', '11715', '11965']
    data = pd.DataFrame(columns=cols)

    for files_path in files_paths:
        dat_file = pd.read_table(files_path, sep='\s+', skiprows=5)
        ti = files_path[65:69]+'/'+files_path[69:71]+'/'+files_path[71:73]+':'+files_path[73:75]
        data.loc[ti] = dat_file.iloc[:, 2].values

    return data
