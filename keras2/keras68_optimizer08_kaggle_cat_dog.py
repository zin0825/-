# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import natsort
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
tf.random.set_seed(111)
np.random.seed(111)
rn.seed(111)



#1. 데이터

# train_datagen = ImageDataGenerator(
#     rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
#     horizontal_flip=True,   # 수평 뒤집기
#     vertical_flip=True,   # 수직 뒤집기
#     width_shift_range=0.1,   # 평행이동
#     height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
#     rotation_range=5,   # 정해진 각도만큼 이미지 회전
#     zoom_range=1.5,   # 축소 또는 화대, 1.2배
#     shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
#     fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,) 

path_train = 'C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/'   
path_test = 'C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path_mission = 'C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

mission_csv = pd.read_csv(path_mission + 'sample_submission.csv', index_col=0)

start1 = time.time()


np_path = 'c:/ai5/_data/_save_npy/'

x_train = np.load(np_path + 'kares43_01_x_train.npy')
y_train = np.load(np_path + 'kares43_01_y_train.npy')
x_test = np.load(np_path + 'kares43_01_x_test.npy') # x트레인 끼리 와이 트레인끼 붙여ㅕㅕㅕ 테스트
y_test = np.load(np_path + 'kares43_01_y_test.npy')   # 위 엑스 트레인, 아래 엑스 트레인 붙이고 엑스 테스트는 밑에서만 활용

# x_train2 = np.load(np_path + 'keras42_01_01_x_train.npy')
# y_train2 = np.load(np_path + 'keras42_01_01_y_train.npy')   
# x_test2 = np.load(np_path + 'keras42_01_01_x_test.npy')
# y_test2 = np.load(np_path + 'keras42_01_01_y_test.npy')   # 이미지 테스트 필요 xxxxx



# x_train = np.concatenate((x_train, x_train2)) 
# y_train = np.concatenate((y_train, y_train2))



x_train = x_train.reshape(x_test.shape[0],
                          x_test.shape[1],
                          x_test.shape[2] * x_train.shape[3])   # 4차원 -> 3차원으로 변경

x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2] * x_test.shape[3])


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(x_train.shape, x_test.shape)



# print(x_train)
# print(x_train.shape)   # (20000, 100, 100, 3) 걸린 시간1 :  0.7 초
# print(y_train)
# print(y_train.shape)   # (20000,) 걸린 시간1 :  0.54 초
# print(x_test)
# print(x_test.shape)   # (100, 3) 걸린 시간1 :  0.55 초
# print(y_test)
# print(y_test.shape)   # (100, 3) 걸린 시간1 :  0.55 초



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=111)



# print(x_train.shape, x_test.shape)  # (35997, 80, 80, 3) (4000, 80, 80, 3)



# augment_size =  10000

# print(x_train.shape[0])   # 35997

# randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)  # [27262  6557  3669 ... 33196 26536 25795]
# print(x_train[0].shape)   # (80, 80, 3)


# x_augmented = x_train[randidx].copy()
# y_augmented = y_train[randidx].copy()
# print(x_augmented.shape, y_augmented.shape)   # (10000, 80, 80, 3) (10000,)


# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],      
#     x_augmented.shape[1],     
#     x_augmented.shape[2], 3)  



# print(x_train.shape, x_test.shape)  # (35997, 80, 80, 3) (4000, 80, 80, 3)

# # # x_augmented = train_datagen.flow(x_augmented, y_augmented,
# # #                                  batch_size=augment_size,
# # #                                  shuffle=False).next()[0]

# print(x_augmented.shape)   # (10000, 80, 80, 3)

# print(x_train.shape, x_test.shape)   # (35997, 80, 80, 3) (4000, 80, 80, 3)

# x_train = np.concatenate((x_train, x_augmented)) 
# y_train = np.concatenate((y_train, y_augmented))


# print(x_train.shape, y_train.shape)   # (45997, 80, 80, 3) (45997,)

# x_train = x_train.reshape(35997, 80, 80, 3)
# x_test = x_test.reshape(4000, 80, 80, 3)


# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
#                                                     test_size=0.1, random_state=313)


# end1 = time.time()

# print('걸린 시간1 : ', round(end1 - start1,2), "초")

# 걸린 시간1 :  9.26 초

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

########## for 문 #############
for learning_rate in lr:

    #2. 모델 구성
    model = Sequential()
    model.add(LSTM(44, input_dim=x_train.shape[1])) 
    model.add(Dense(44))
    model.add(Dense(40))
    model.add(Dropout(0.25))
    model.add(Dense(40))
    model.add(Dense(30))
    model.add(Dense(12))
    model.add(Dense(1, activation='softmax'))


    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))   
    # learning_rate=learning_rate 숫자 장난. 0.01로 넣어줘도 됨
    # learning_rate 디폴트 0.001

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=8,
          batch_size=4, 
          verbose=0
          )




    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))



# #5. 파일 출력
# y_submit = model.predict(y_test, batch_size=12)
# mission_csv['label'] = y_submit
# print(mission_csv)

# # mission_csv.to_csv(path_mission + 'mission_0806_1738.csv')



# 걸린 시간1 :  0.33 초
# acc :  1.0
# 걸린 시간 :  0.0 초
# 로스 :  [3.3390790800224636e-23, 1.0]



# augment
# 걸린 시간1 :  3.51 초
# 0.7532608695652174
# acc :  0.7532608695652174
# 걸린 시간 :  481.71 초
# 로스 :  [0.5587460994720459, 0.7532608509063721]


# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 4.6453328132629395
# lr : 0.1, r2 : -0.0010024249614578973
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 4.609957695007324
# lr : 0.01, r2 : -0.00016506354769926345
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 4.608075141906738
# lr : 0.005, r2 : -0.00012576056122785785
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 4.6076788902282715
# lr : 0.001, r2 : -0.00011660689443381721
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 3.4975950717926025
# lr : 0.0005, r2 : 0.06963075564320818
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 3.5402746200561523
# lr : 0.0001, r2 : 0.06896193273731606


# ??????
# Traceback (most recent call last):
#   File "c:\프로그램\ai5\study\keras\keras49_augment5_cat_dog.py", line 228, in <module>
#     mission_csv['label'] = y_submit
#   File "c:\Users\jin\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\pandas\core\frame.py", line 3612, in __setitem__
#     self._set_item(key, value)
#   File "c:\Users\jin\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\pandas\core\frame.py", line 3784, in _set_item
#     value = self._sanitize_column(value)
#   File "c:\Users\jin\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\pandas\core\frame.py", line 4509, in _sanitize_column
#     com.require_length_match(value, self.index)
#   File "c:\Users\jin\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\pandas\core\common.py", line 531, in require_length_match
#     raise ValueError(
# ValueError: Length of values (3000) does not match length of index (12500)

# LSTM

# k59_18_


