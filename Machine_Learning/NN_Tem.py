

import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from Extract_Observation_bt import Extract_observation_bt
print(tf.__version__)



# 获取数据
Years = ['2007', '2009', '2010']
merge_dataset = Extract_observation_bt(Years)
# 将'Rain'列转换为离散数值
merge_dataset['Rain'] = pd.Categorical(merge_dataset['Rain'])
merge_dataset['Rain'] = merge_dataset.Rain.cat.codes
# 去除空数据
merge_dataset[merge_dataset < - 1000] = np.nan
merge_dataset.dropna(inplace=True)

Summer = ['05', '06', '07', '08', '09']
for i in merge_dataset.index:
    if(i[5:7]  in Summer):
        merge_dataset.drop(i, inplace=True)
merge_dataset


# 数据筛选
# '10', 'Tamb(K)', 'Rh(%)', 'Pres(mb)', 'Tir(K)', 'Rain', 'Azim', 'Elev', 'TkBB(K)'
merge_dataset.drop(['10', 'Rain', 'TkBB(K)', 'Rh(%)', 'Pres(mb)', 'Tir(K)', 'Azim', 'Elev',], axis=1, inplace=True)
# ' 22.235', ' 23.035', ' 23.835', ' 26.235', ' 30.000', ' 51.250', ' 52.280', ' 53.850', ' 54.940', ' 56.660', ' 57.290', ' 58.800'
merge_dataset.drop([' 22.235', ' 23.035', ' 23.835', ' 26.235', ' 30.000'], axis=1, inplace=True)
# '0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00',
# '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75',
# '4.00', '4.25', '4.50', '4.75', '5.00', '5.25', '5.50', '5.75', '6.00', '6.25', '6.50',
# '6.75', '7.00', '7.25', '7.50', '7.75', '8.00', '8.25', '8.50', '8.75', '9.00', '9.25',
# '9.50', '9.75', '10.00'

# 保留小数位
merge_dataset = merge_dataset.round(1)
# 归一化
def norm(x):
    return (x - merge_dataset.describe().transpose()['mean']) / merge_dataset.describe().transpose()['std']
merge_dataset.iloc[:, :8] = norm(merge_dataset).iloc[:, :8]



# 乱序，分离训练数据与测试数据
train_data = merge_dataset.sample(frac=0.8, random_state=0)
test_data = merge_dataset.drop(train_data.index)
# 分离标签
train_labels = train_data.iloc[:, 8:]
test_labels = test_data.iloc[:, 8:]
train_data.drop(train_labels.columns, axis=1, inplace=True)
test_data.drop(train_labels.columns, axis=1, inplace=True)
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)



def build_model():
    My_model = keras.Sequential([
        layers.Dense(len(train_data.keys()), activation='relu', input_shape=[len(train_data.keys())]),
        layers.Dense(47, activation='relu'),
        layers.Dense(47)
    ])

    opt = tf.optimizers.Adam(1e-3)
    # tf.train.GradientDescentOptimizer(0.001)
    # tf.keras.optimizers.RMSprop(0.001)
    # tf.optimizers.Adam(1e-3)

    My_model.compile(loss='mse',
                     optimizer=opt,
                     metrics=['mae', 'mse']
                     )
    return My_model
model = build_model()
model.summary()



history = model.fit(train_data,
                    train_labels,
                    batch_size=32,
                    epochs=2000,
                    verbose=0,
                    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch



def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 10])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 70])
    plt.legend()
    plt.show()
plot_history(history)



loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)

print("MAE: {:5.2f} ".format(mae))
print("MSE: {:5.2f} ".format(mse))
print("LOSS: {:5.2f} ".format(loss))



r1 = 0
for i in range(10):
    r = np.corrcoef(model.predict(test_data[i:i + 1]), test_labels[i:i + 1])
    r1 += r[0, 1]
# print(r1/10)
print('相关系数 {:.4}'.format(r1 / 10))



for i in range(20):
    flag = random.randrange(0, 200)
    fig = plt.figure(figsize=(4, 10))
    Height = test_labels.columns
    X1 = model.predict(test_data[flag:flag + 1])
    X2 = test_labels[flag:flag + 1].T
    plt.plot(np.abs(X1 - X2), Height)
    # plt.plot(X1, Height)
    # plt.plot(X2, Height)
    plt.show()
    print('相关系数{:.4}'.format(np.corrcoef(X1.T, X2.T)[0, 1]))
    # print('平均偏差{:.4}'.format((X1-X2).mean().values))
    # print(np.corcoef(X1.T,X2.T))
    print(np.abs(np.mean(X1 - X2).round(4)))