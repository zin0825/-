

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)   # 샘플만 요약해 나와서 000만 나옴
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  6만의 28, 28, 1 / 컬러는 6만의 28, 28, 3
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)   만의 28, 28, 1

# x는 6만의 28, 28, 1로 리쉐이프
# y는 원핫앤코딩



x_train = x_train.reshape(60000, 28, 28, 1)   # 1이 없으면 2차원이니까 3차원으로 정의하기 위해
x_test = x_test.reshape(10000, 28, 28, 1)



y_train = pd.get_dummies(y_train)   # 이걸 해야 10이 생김
y_test = pd.get_dummies(y_test)
print(y_train)   # [60000 rows x 10 columns]
print(y_test)   # [10000 rows x 10 columns]

print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)

#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1)))   # input_shape 3차원 형태, 가로, 세로, 컬러 (2,2)로 조각 내겠다 # 28,28,1 -> 27,27,10
                            # shape = (batch_size, rows, columns, channels)   # 왜 batch_size인가 훈련의 데이터를 배치사이즈 단위로 한다.
                                                                              # 행의 갯수는 중요하지 않음 전체에서 얼마만큼 잘라 쓰느냐
                            # shape = (batch_size, heights, widths, channels)   



model.add(Conv2D(filters=20, kernel_size=(3,3)))   # 커널사이즈 3,3로 자른다 / 공식적인 파라미터 명칭은 필터 # 3,3 -> 25,25,25

model.add(Conv2D(15, (4,4)))   # (N,22, 22, 15) -> 같으려면 (Flatten) (N, 22 x 22 x 15) 모양 (shape)만 바뀐것
model.add(Flatten()) # 얘가 없으면 덴스를 못 씀. 2D와 덴트(다차원)을 연결하기 위해 반드시 필요

model.add(Dense(units=8))
model.add(Dense(units=9, input_shape=(8,)))
                         # shape = (batch_size, input_dim)

model.add(Dense(10, activation='softmax'))   # 0부터 9까지 y를 맞추기 위해

# model.summary()
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #
# # =================================================================
# #  conv2d (Conv2D)             (None, 27, 27, 10)        50

# #  conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820

# #  conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815

# #  flatten (Flatten)           (None, 7260)              0   # 왜 0인가? 모양만 (펴준기만) 바뀐거라 0이다

# #  dense (Dense)               (None, 8)                 58088

# #  dense_1 (Dense)             (None, 9)                 81

# # =================================================================
# # Total params: 64,854
# # Trainable params: 64,854
# # Non-trainable params: 0


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
                   restore_best_weights=True)

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   # 현재 시간
print(date)   # 2024-07-26 16:50:13.613311
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
print(type(date))



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_01',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)   # 파일네임, 패스 더하면 요놈

start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=70,   # 에포 300 ->10 배치 32 ->70 변경
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp],   # 두개 이상은 리스트
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')   # 두가지 다 저장할거야

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)   # 회귀가 아니기 때문에 acc가 더 정확함
# print('r2 score : ', r2)

accuracy_score = accuracy_score(y_test, y_predict)

print("걸린 시간 : ", round(end - start,2),'초')


# cpu
# 로스 :  0.18052002787590027
# r2 score :  -1.0270623603944324
# 걸린 시간 :  174.5 초

# gpu
# 로스 :  0.15059642493724823
# r2 score :  -0.661165124426
# 걸린 시간 :  26.72 초


# 에포 300 ->10 배치 32 ->70 변경
# cpu
# 로스 :  0.17936007678508759
# 걸린 시간 :  159.7 초

# gpu
# 로스 :  0.18084010481834412
# 걸린 시간 :  24.57 초