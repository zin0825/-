# Conv2D로 시작해서 중간에



import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터

# x_train = x_train/255.
# x_test = x_test/255.   # 이거 할거면 밑에거 지워야함

# train_datagen = ImageDataGenerator(
#     # rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
#     horizontal_flip=True,   # 수평 뒤집기
#     vertical_flip=True,   # 수직 뒤집기
#     width_shift_range=0.2,   # 평행이동
#     # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
#     rotation_range=15,   # 정해진 각도만큼 이미지 회전
#     # zoom_range=1.2,   # 축소 또는 화대, 1.2배
#     # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
#     fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
# )





(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)   # 샘플만 요약해 나와서 000만 나옴
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  6만의 28, 28, 1 / 컬러는 6만의 28, 28, 3
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)   만의 28, 28, 1


# ##### 스켈일링 1-1
x_train = x_train/255.   # 소수점
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0


### 원핫 y 1-1 케라스 // 판다스, 사이킷런으로도 맹그러  항상 시작은 0부터다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)


#2. 모델
# model = Sequential()
# model.add(Dense(28, input_shape=(28,28)))   # (n, 28, 28)
# model.add(Reshape(target_shape=(28,28,1)))   # (n, 28, 28, 1)
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))  # 26, 26, 64
# model.add(Flatten())   # 23 * 23 * 32

# model.add(Dense(100, input_shape=(28,28)))   # (n, 28, 28)
# model.add(Reshape(target_shape=(28,100,1)))   # (n, 28, 28, 1)
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))  # 26, 26, 64
# model.add(Flatten())   # 23 * 23 * 32

model = Sequential()
model.add(Dense(280, input_shape=(28, 28)))   # (n, 28, 28)   # 위에서 리쉐잎한걸 밑에서 리쉐잎 가능 덴스는 다차원도 들어갈수 있음 현재 3차원
model.add(Reshape(target_shape=(28, 28, 10)))   # (n, 28, 28, 1)  여깄는 숫자가 위에 덴스 값에 맞춰야함 곱해서 맞기만 하면 됨 
# LSTM 넣을 때는 3차원으로 넣어야함 / # conv2D는 4차원으로 넣어야함. 앞에 none은 생략하고 3차원 쉐잎으로 리쉐잎함 (써머리로 확인가능)
model.add(Conv2D(64, (3,3)))   # 26, 26, 64    얘는 4차원
model.add(MaxPooling2D())     # 13, 13, 64   2분의 1로 나눔
model.add(Conv2D(5, (4,4),))   # 10, 10, 5

# model.add(Reshape(target_shape=(10 * 10, 5)))    
# model.add(Reshape(target_shape=(10 * 10 * 5,)))   # (500,) = (10, 10, 5)와 같다 /  벡터 형태이기 때문에 , 넣어줘야함 
model.add(Reshape(target_shape=(500,)))   # (500,) = (10 * 10 * 5)와 같다 /  벡터 형태이기 때문에 , 넣어줘야함 

model.add(Flatten())   

model.add(Dense(units=32))   # dense는 2차원이기 때문에 앞에 none을 생략하고 1차원으로 리쉐잎함 그래서 앞에서 다 곱하고 , 붙임
model.add(Dense(10, activation='softmax'))   # 0부터 9까지 y를 맞추기 위해

model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 28, 280)           8120

#  reshape (Reshape)           (None, 28, 28, 10)        0

#  conv2d (Conv2D)             (None, 26, 26, 64)        5824

#  max_pooling2d (MaxPooling2D  (None, 13, 13, 64)       0
#  )

#  conv2d_1 (Conv2D)           (None, 10, 10, 5)         5125

#  reshape_1 (Reshape)         (None, 500)               0

#  flatten (Flatten)           (None, 500)               0

#  dense_1 (Dense)             (None, 32)                16032

#  dense_2 (Dense)             (None, 10)                330








#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])   # metrics 정확도 확인

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



path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k35_04',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
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
hist = model.fit(x_train, y_train, epochs=5, batch_size=1228,   
          verbose=1, 
          validation_split=0.2,
          callbacks=[es, mcp],   # 두개 이상은 리스트
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')   # 두가지 다 저장할거야

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)
print('acc : ', round(loss[1],2))

y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)   # 회귀가 아니기 때문에 acc가 더 정확함
# print('r2 score : ', r2)


y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)   # 원핫인코딩한 애들을 다시 원핫인코딩 하기 전으로 변환

print(y_predict)

accuracy = accuracy_score(y_test, y_predict)   # 변수에 (초기화 안된) 변수를 넣어서 오류 뜸 
acc = accuracy_score(y_test, y_predict) # 예측한 y값과 비교
print("acc_score : ", acc)
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

# 에포 1
# acc_score :  0.8979
# 걸린 시간 :  4.77 초


# 로스 :  [0.3007262349128723, 0.9153000116348267]
# acc :  0.92
# acc_score :  0.9153
# 걸린 시간 :  21.72 초


# model-reshape
# acc_score :  0.954
# 걸린 시간 :  48.89 초

# acc_score :  0.956
# 걸린 시간 :  48.5 초











