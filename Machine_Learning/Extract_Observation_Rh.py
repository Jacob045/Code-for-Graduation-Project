#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/29 19:47
# @File     : Extract_Observation_Tem.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com
import os
import pandas as pd
import numpy as np


def gongshi(T, Td):
    a = 17.27
    b = 237.7
    m1 = (a*b)/(b+T)*Td - (a*b*T)/(b+T)
    m2 = m1/(b+Td)
    return 100*np.exp(m2)


def extract_observation_rh(years):
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
    dat1 = pd.DataFrame(columns=cols)

    for files_path in files_paths:
        dat_file = pd.read_table(files_path, sep='\s+', skiprows=5)
        ti = files_path[65:69]+'/'+files_path[69:71]+'/'+files_path[71:73]+':'+files_path[73:75]
        dat1.loc['Dp_tem'] = dat_file.iloc[:, 3].values
        dat1[dat1 < -9000] = 0.001
        dat1.loc['tem'] = dat_file.iloc[:, 2].values
        if dat1.loc['tem'].mean() < 0:
            continue
        dat1.loc['Rh'] = gongshi(dat1.loc['tem'], dat1.loc['Dp_tem'])
        data.loc[ti] = dat1.loc['Rh']
        # data.loc[ti] = dat_file.iloc[:, 2].values

    return data

