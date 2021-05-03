#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/16 19:50
# @File     : Extract_Simulated_bt.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com

import pandas as pd
import os


def extract_simulated_bt(years):
    dir_path = r'E:\test\test'
    files_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file[30:34] in years:
                files_paths.append(os.path.join(root, file))
    cols = ['22.235', '23.035', '23.835', '26.235', '30.000', '51.250',
            '52.280', '53.850', '54.940', '56.660', '57.290', '58.800']
    data = pd.DataFrame(columns=cols)

    for files_path in files_paths:
        dat_file = pd.read_csv(files_path, sep='\s+', index_col='Time')
        data = data.append(dat_file)

    def trans(i):
        i = str(i)
        return i[0:4] + '/' + i[4:6] + '/' + i[6:8] + ':' + i[8:10]
    data.index = [trans(i) for i in data.index]

    return data
