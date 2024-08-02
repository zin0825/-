"""
01. 보스톤
02. california
03. diabetes
04. dacon_ddarung
05. kaggle_bike

06_cancer
07_dacon_diabetes
08_kaggle_bank
09_wine
10_fetch_covtpe
11_digits
"""


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


print(tf.__version__)   # 2.7.4


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

    


#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (20640, 8) (20640,)

x = x.reshape(20640, 8, 1, 1)   # 1, 1 내 맘대로

x = x/255.

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)


#2. 모델구성
model = Sequential()
model.add(Conv2D(300, (2,1), input_shape=(8, 1, 1), activation='relu', strides=1, padding='same'))
model.add(Conv2D(filters=300, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(Conv2D(200, (3,3), activation='relu', strides=1, padding='same'))

model.add(Flatten())

model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
                   restore_best_weights=True)

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   
print(date)  
print(type(date))  
date = date.strftime("%m%d_%H%M")   
print(date)   
print(type(date))



path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   


filepath = "".join([path, 'k39_02',date, '_' , filename]) 


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath)    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=916,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('acc : ', round(loss[1],2))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

print("걸린시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


# 로스 :  0.5418245196342468
# r2스코어 :  0.6115782798620845
# 걸린시간 :  30.78 초
# 쥐피유 돈다!!!


# 로스 :  0.5869264602661133
# r2스코어 :  0.5792457153915529
# 걸린시간 :  18.28 초

# dnn -> cnn
# 로스 :  [0.4840528964996338, 0.0007267441833391786]
# r2스코어 :  0.6529935210645953
# 걸린시간 :  15.05 초