import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f'tf.version is {tf.__version__}')

# 08时数据
lv1_08 = pd.read_csv(
    r'I:\Data\Personal Data\graduation project\Code\Machine Learning\basicdata\2007_lv1_08.csv',
    index_col='Unnamed: 0')
Temperature_08 = pd.read_csv(
    r'I:\Data\Personal Data\graduation project\Code\Machine Learning\basicdata\2007_Temperature_08.csv',
    index_col='Unnamed: 0')
merge_dataset = pd.merge(lv1_08, Temperature_08, on=lv1_08.index)
merge_dataset.drop('key_0', axis=1, inplace=True)
# print(merge_dataset.shape)

# 20时数据
lv1_08 = pd.read_csv(
    r'I:\Data\Personal Data\graduation project\Code\Machine Learning\basicdata\2007_lv1_20.csv',
    index_col='Unnamed: 0')
Temperature_08 = pd.read_csv(
    r'I:\Data\Personal Data\graduation project\Code\Machine Learning\basicdata\2007_Temperature_20.csv',
    index_col='Unnamed: 0')
merge_dataset1 = pd.merge(lv1_08, Temperature_08, on=lv1_08.index)
merge_dataset1.drop('key_0', axis=1, inplace=True)

# 连接
merge_dataset = merge_dataset.append(merge_dataset1)

# 将'Rain'列转换为离散数值
merge_dataset['Rain'] = pd.Categorical(merge_dataset['Rain'])
merge_dataset['Rain'] = merge_dataset.Rain.cat.codes

# 去除空数据
merge_dataset[merge_dataset < -1000] = np.nan
merge_dataset.dropna(inplace=True)

# 温度训练
merge_dataset.drop(
    ['10', 'Rain', 'TkBB(K)', 'Rh(%)', 'Pres(mb)', 'Tir(K)', 'Azim', 'Elev', ' 22.235', ' 23.035', ' 23.835', ' 26.235',
     ' 30.000'], axis=1, inplace=True)
merge_dataset = merge_dataset.round(1)
# print(merge_dataset.columns)

# 归一化
def norm(x):
    return (x - merge_dataset.describe().transpose()['mean']) / merge_dataset.describe().transpose()['std']


merge_dataset.iloc[:, :8] = norm(merge_dataset).iloc[:, :8]

# 乱序，分离训练数据与测试数据
train_data = merge_dataset.sample(frac=0.8, random_state=0)
test_data = merge_dataset.drop(train_data.index)
# 分离数据和标签
train_labels = train_data.iloc[:, 8:]
test_labels = test_data.iloc[:, 8:]
train_data.drop(train_labels.columns, axis=1, inplace=True)
test_data.drop(train_labels.columns, axis=1, inplace=True)

print(
    f'train_data.shape is {train_data.shape}, '
    f'test_data.shape is {test_data.shape}, '
    f'train_labels.shape is {train_labels.shape}, '
    f'test_labels.shape is {test_labels.shape} '
)


def build_model():
    model = keras.Sequential([
        layers.Dense(len(train_data.keys()), activation='relu', input_shape=[len(train_data.keys())]),
        layers.Dense(23, activation='relu'),
        layers.Dense(47)
    ])

    opt = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mae', 'mse']
                  )
    return model


model = build_model()
model.summary()

history = model.fit(train_data,
                    train_labels,
                    batch_size=32,
                    epochs=1500,
                    verbose=2,
                    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 4])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 10])
    plt.legend()
    plt.show()


# plot_history(history)

loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)

print("MAE: {:5.2f} ".format(mae))
print("MSE: {:5.2f} ".format(mse))
print("LOSS: {:5.2f} ".format(loss))

r1 = 0
for i in range(10):
    r = np.corrcoef(model.predict(train_data[i:i+1]), test_labels[i:i+1])
    r1 += r[0,1]
# print(r1/10)
print('相关系数 {:.4}'.format(r1/10))

flag = 2
fig = plt.figure(figsize=(4,10))
Height = test_labels.columns
X1 = model.predict(test_data[flag:flag+1]).T
X2 = test_labels[flag:flag+1].T
plt.plot(np.abs(X1-X2),Height)
# plt.plot(X1,Height)
# plt.plot(X2,Height)
plt.show()
print('相关系数{:.4}'.format(np.corrcoef(X1.T,X2.T)[0,1]))
# print('平均偏差{:.4}'.format((X1-X2).mean().values))
# print(np.corrcoef(X1.T,X2.T))
print(np.abs(np.mean(X1-X2).round(4)))
