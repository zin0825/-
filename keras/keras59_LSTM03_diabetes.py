import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (442, 10) (442,)

x = x.reshape(442, 10, 1, 1)

x = x/255.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)



#2. 모델구성
model = Sequential()
model.add(LSTM(200, input_shape=(10,1))) 
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))



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



path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   


filepath = "".join([path, 'k59_03_01_',date, '_' , filename]) 


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


# 로스 :  2312.763916015625
# r2스코어 :  0.5750138480417163
# 걸린시간 :  2.54 초
# 쥐피유 없다! xxxxx

# 로스 :  2490.44970703125
# r2스코어 :  0.5423628463117105
# 걸린시간 :  7.8 초
# 쥐피유 돈다!!!

# dnn -> cnn
# r2스코어 :  -0.07379311573165492
# 걸린시간 :  4.25 초
# 로스 :  [5843.55517578125, 0.0]


# LSTM
# r2스코어 :  -0.04227134682976308
# 걸린시간 :  4.64 초
# 로스 :  [5672.01416015625, 0.0]
# k59_03_

# r2스코어 :  -0.02525805852789209
# 걸린시간 :  3.78 초
# 로스 :  [5579.4287109375, 0.0]
# k59_03_01_
