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
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)

print(x)
print(y)
print(x.shape, y.shape)   # (20640, 8) (20640,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0000000000000004
print(np.max(x_test), np.max(x_test))   # 1.2491334943808423 1.2491334943808423




# #2. 모델구성
# model = Sequential()
# model.add(Dense(30, activation='relu', input_dim=8))
# model.add(Dropout(0.3)) 
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='linear'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(8,))
dense1 = Dense(30, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(30, name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(20, name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(20, name='ys4')(drop3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 8)]               0

#  ys1 (Dense)                 (None, 30)                270

#  dropout (Dropout)           (None, 30)                0

#  ys2 (Dense)                 (None, 30)                930

#  dropout_1 (Dropout)         (None, 30)                0

#  ys3 (Dense)                 (None, 20)                620

#  dropout_2 (Dropout)         (None, 20)                0

#  ys4 (Dense)                 (None, 20)                420

#  dense (Dense)               (None, 1)                 21

# =================================================================
# Total params: 2,261
# Trainable params: 2,261
# Non-trainable params: 0



# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# start = time.time()

# from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_loss',
#                    mode= 'min',
#                    patience=10,
#                    verbose=1, 
#                    restore_best_weights=True)

# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# import datetime   # 날짜
# date = datetime.datetime.now()   # 현재 시간
# print(date)   # 2024-07-26 16:50:13.613311
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
# print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
# print(type(date))



# path = './_save/keras32_mcp2/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# filepath = "".join([path, 'k32_02',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )   # 파일네임, 패스 더하면 요놈
 


# hist = model.fit(x_train, y_train, epochs=100, batch_size=64, 
#           verbose=1, validation_split=0.3,
#           callbacks=[es, mcp])
# end = time.time()



# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print("로스 : ", loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print("r2스코어 : ", r2)

# print("걸린시간 : ", round(end - start, 2), "초")


# 로스 :  0.31834354996681213
# r2스코어 :  0.7717867918008543
# 걸린시간 :  17.23 초

# propout
# 로스 :  0.5241807699203491
# r2스코어 :  0.6242267750746597
# 걸린시간 :  5.19 초