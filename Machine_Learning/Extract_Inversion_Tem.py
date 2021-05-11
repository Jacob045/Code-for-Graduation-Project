#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/5/4 15:48
# @File     : Extract_Inversion_Tem.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com
import search_filepth_module
import pandas as pd

def extract_inversion_tem(years):
    cols = [' 0.00', ' 0.10', ' 0.20', ' 0.30', ' 0.40', ' 0.50', ' 0.60', ' 0.70',
            ' 0.80', ' 0.90', ' 1.00', ' 1.25', ' 1.50', ' 1.75', ' 2.00', ' 2.25',
            ' 2.50', ' 2.75', ' 3.00', ' 3.25', ' 3.50', ' 3.75', ' 4.00', ' 4.25',
            ' 4.50', ' 4.75', ' 5.00', ' 5.25', ' 5.50', ' 5.75', ' 6.00', ' 6.25',
            ' 6.50', ' 6.75', ' 7.00', ' 7.25', ' 7.50', ' 7.75', ' 8.00', ' 8.25',
            ' 8.50', ' 8.75', ' 9.00', ' 9.25', ' 9.50', ' 9.75', '10.00']

    ouput_data = pd.DataFrame(columns=cols)

    microwave_dirpath = r'I:\Data\Personal_Data\graduation_project\SACOL\microwave\\' + years[0]
    microwave_target_str = 'lv2'
    microwave_filepaths = search_filepth_module.search_filepath(microwave_dirpath, microwave_target_str)

    for microwave_filepath in microwave_filepaths:
        data = pd.read_csv(microwave_filepath, index_col='Record')
        data = data.loc[data['10'] == 11]
        data.drop(data.columns[1:9], axis=1, inplace=True)
        data = data.reset_index(drop=True)

        num0800 = num0830 = num2000 = num2030 = 0
        for i in data.index:
            x = data.loc[i, 'Date/Time'][9:14]
            if x == '08:00':
                num0800 = i
            elif x == '08:30':
                num0830 = i
            elif x == '20:00':
                num2000 = i
            elif x == '20:30':
                num2030 = i

        date = microwave_filepath[70:80].replace('-', '/')
        if (num0800 != 0) and (num0830 != 0):
            ouput_data.loc[date+':08', :] = data.iloc[num0800:num0830+1, ].mean()
        if (num2000 != 0) and (num2030 != 0):
            ouput_data.loc[date+':20', :] = data.iloc[num2000:num2030+1, ].mean()

    return ouput_data
