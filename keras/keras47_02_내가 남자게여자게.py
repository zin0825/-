import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import natsort




#1. 데이터

np_path = 'C:\\프로그램\\ai5\\_data\\image\\me\\'


x_test = np.load(np_path + 'me_arr.npy')

model = load_model('C:\\프로그램\\ai5\\_save\\keras45\\k45_08_0805_1620_0007-0.5844.hdf5')

y_pred = model.predict(x_test)

print(y_pred)
# [[0.55793005]]   # 55프로의 확률로 1

y_pred = np.round(y_pred)   # 반올림
print(y_pred)
# [[1.]]

# 0에서 1사이 값
# 0.3이면 70프로 확률로 0이고,  0.7이면 70프로 확률로 1이다.

print('나는 {}의 확률로 여자다.'.format(y_pred))
