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

start1 = time.time()

x_test = np.load(np_path + 'me_arr.npy')

model = load_model('C:\\프로그램\\ai5\\_save\\keras42\\k42_02_0805_1412_0009-0.0000.hdf5')

y_pred = model.predict(x_test)

print(y_pred)

# [[0.]] -> 개

print('나는 {}의 확률로 개다.'.format(y_pred))




