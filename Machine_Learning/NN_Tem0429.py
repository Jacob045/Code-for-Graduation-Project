#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jacob045
# @Time     : 2021/4/29 21:16
# @File     : NN_Tem0429.py
# @Project  : Code
# @E-mail   : Jacob045@foxmial.com

import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from Extract_Observation_Tem import extract_observation_tem
from Extract_Simulated_bt import extract_simulated_bt
print(tf.__version__)


# 获取模拟亮温数据
Years = ['1997', '1998', '1999', '2000', '2001', '2002', '2003',
         '2004', '2005', '2006', '2007', '2008', '2009', '2010']
train_data = extract_simulated_bt(Years)
train_data = train_data.round(1)
print(train_data.shape)


# 模拟亮温数据归一化
def norm(x):
    return (x - x.describe().transpose()['mean']) / x.describe().transpose()['std']
train_data = norm(train_data)


# 模拟亮温数据拆分
test_data = train_data.sample(frac=0.2, random_state=1)
train_data.drop(test_data.index, inplace=True)
train_data = train_data.sample(frac=1)
print(f'train_data shape is {train_data.shape}')
print(f'test_data shape is {test_data.shape}')


# 获取观测温度数据
train_labels = extract_observation_tem(Years)

# 拆分观测温度数据
test_labels = train_labels.loc[test_data.index]
train_labels = train_labels.loc[train_data.index]
print(f'train_labels shape is {train_labels.shape}')
print(f'test_labels shape is {test_labels.shape}')



def build_model():
    My_model = keras.Sequential([
        layers.Dense(len(train_data.keys()), activation='relu', input_shape=[len(train_data.keys())]),
        layers.Dense(47, activation='relu'),
        layers.Dense(len(train_labels.keys()))
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
                    verbose=2,
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
# plot_history(history)



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
    X1 = model.predict(test_data[flag:flag + 1]).T
    X2 = test_labels[flag:flag + 1].T
    plt.plot(np.abs(X1 - X2), Height)
    # plt.plot(X1, Height)
    # plt.plot(X2, Height)
    plt.show()
    print('相关系数{:.4}'.format(np.corrcoef(X1.T, X2.T)[0, 1]))
    # print('平均偏差{:.4}'.format((X1-X2).mean().values))
    # print(np.corcoef(X1.T,X2.T))
    print(np.abs(np.mean(X1 - X2).round(4)))
    print('\n')