import pandas as pd

def Load_Data():
    merge_1 = pd.DataFrame()
    Years = ['2007', '2009', '2010']
    Times = ['08', '20']
    Path = r'I:\Data\Personal_Data\graduation_project\Code\Machine_Learning\Basic_Data\\'
    for Time in Times:
        for Year in Years:
            lv1 = pd.read_csv(
                Path + Year + '_lv1_' + Time + '.csv',
                index_col='Unnamed: 0')
            Temperature = pd.read_csv(
                Path + Year + '_Temperature_' + Time + '.csv',
                index_col='Unnamed: 0')
            # 横向合并
            merge_0 = pd.merge(lv1, Temperature, on=lv1.index).set_index(['key_0'])
            # 修改index
            merge_0.index = [i+':'+Time for i in merge_0.index]
            # 合并到merge_1，纵向合并
            merge_1 = merge_1.append(merge_0)
    return merge_1