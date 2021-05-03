import pandas as pd
import os


def extract_observation_bt(years):
    merge_1 = pd.DataFrame()
    times = ['08', '20']
    path = r'I:\Data\Personal_Data\graduation_project\Code\Machine_Learning\Basic_Data\\'
    for time in times:
        for year in years:
            lv1 = pd.read_csv(
                path + year + '_lv1_' + time + '.csv',
                index_col='Unnamed: 0')
            temperature = pd.read_csv(
                path + year + '_Temperature_' + time + '.csv',
                index_col='Unnamed: 0')
            # 横向合并
            merge_0 = pd.merge(lv1, temperature, on=lv1.index).set_index(['key_0'])
            # 修改index
            merge_0.index = [i+':'+time for i in merge_0.index]
            # 合并到merge_1，纵向合并
            merge_1 = merge_1.append(merge_0)
    return merge_1

