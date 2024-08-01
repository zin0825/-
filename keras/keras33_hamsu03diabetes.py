import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_diabetes()
print(datasets)
print(datasets.DESCR)   # describe 확인 
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)

print(x)
print(y)
print(x.shape, y.shape)   # (442, 10) (442,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0  # 이건 랜덤값에 따라 0이 나올수도 -가 나올 수도 있다. 
# 랜덤 33 - 0.0 1.0  랜덤 9 - -어쩌구



# #2. 모델구성
# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=10))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='linear'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(10,))
dense1 = Dense(50, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(50, name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(50, name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(20, name='ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(10, name='ys5')(drop4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 10)]              0

#  ys1 (Dense)                 (None, 50)                550

#  dropout (Dropout)           (None, 50)                0

#  ys2 (Dense)                 (None, 50)                2550

#  dropout_1 (Dropout)         (None, 50)                0

#  ys3 (Dense)                 (None, 50)                2550

#  dropout_2 (Dropout)         (None, 50)                0

#  ys4 (Dense)                 (None, 20)                1020

#  dropout_3 (Dropout)         (None, 20)                0

#  ys5 (Dense)                 (None, 10)                210

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 6,891
# Trainable params: 6,891
# Non-trainable params: 0
# ________________________________


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
# filepath = "".join([path, 'k32_03',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )


# hist = model.fit(x_train, y_train, epochs=100, batch_size=3, 
#           verbose=1, validation_split=0.3,
#           callbacks=[es, mcp])
# end = time.time()





# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print("로스 : ", loss)

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# print("r2스코어 : ", r2)


# 로스 :  2368.745849609375
# r2스코어 :  0.5647267605952928


# # Dropout
# 로스 :  2540.94775390625
# r2스코어 :  0.5330834988671461