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


import tensorflow as tf



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

x_train = np.load(np_path + 'keras41_03_01_x_train.npy')
y_train = np.load(np_path + 'keras41_03_01_y_train.npy')
x_test = np.load(np_path + 'keras41_03_01_x_test.npy') # x트레인 끼리 와이 트레인끼 붙여ㅕㅕㅕ 테스트
y_test = np.load(np_path + 'keras41_03_01_y_test.npy')   # 위 엑스 트레인, 아래 엑스 트레인 붙이고 엑스 테스트는 밑에서만 활용

x_train2 = np.load(np_path + 'keras42_01_01_x_train.npy')
y_train2 = np.load(np_path + 'keras42_01_01_y_train.npy')   
# x_test2 = np.load(np_path + 'keras42_01_01_x_test.npy')
# y_test2 = np.load(np_path + 'keras42_01_01_y_test.npy')   # 이미지 테스트 필요 xxxxx



x_train = np.concatenate((x_train, x_train2)) 
y_train = np.concatenate((y_train, y_train2))





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
end1 = time.time()


print(x_train.shape, x_test.shape)  # (35997, 80, 80, 3) (4000, 80, 80, 3)



augment_size =  10000

print(x_train.shape[0])   # 35997

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)  # [27262  6557  3669 ... 33196 26536 25795]
print(x_train[0].shape)   # (80, 80, 3)


x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)   # (10000, 80, 80, 3) (10000,)


x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)  



print(x_train.shape, x_test.shape)  # (35997, 80, 80, 3) (4000, 80, 80, 3)

# # x_augmented = train_datagen.flow(x_augmented, y_augmented,
# #                                  batch_size=augment_size,
# #                                  shuffle=False).next()[0]

print(x_augmented.shape)   # (10000, 80, 80, 3)

print(x_train.shape, x_test.shape)   # (35997, 80, 80, 3) (4000, 80, 80, 3)

x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))


print(x_train.shape, y_train.shape)   # (45997, 80, 80, 3) (45997,)

x_train = x_train.reshape(35997, 80, 80, 3)
x_test = x_test.reshape(4000, 80, 80, 3)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                    test_size=0.1, random_state=313)


end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1,2), "초")

# 걸린 시간1 :  9.26 초


#2. 모델 구성
model = Sequential()
model.add(LSTM(44, input_shape=(80, 80 * 3))) 
model.add(Dense(44))
model.add(Dense(40))
model.add(Dropout(0.25))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(12))
model.add(Dense(1, activation='softmax'))



# model.add(Conv2D(44, (2,2), input_shape=(80, 80, 3), activation='relu',
#                  strides=1, padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=44, kernel_size=(2,2), activation='relu', 
#                  strides=1, padding='same'))
# model.add(Conv2D(40, (2,2), strides=1, padding='same'))
# model.add(Dropout(0.25))
# model.add(BatchNormalization())
# model.add(Conv2D(40, (2,2), strides=1, padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(30, (2,2), strides=1, padding='same'))
# model.add(Flatten())

# model.add(Dense(80, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)


######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))  
date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k59_18_01_', date, '_', filename])  


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)


start = time.time()
hist = model.fit(x_train, y_train, epochs=80, batch_size=4,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])

end = time.time()



#4. 평가, 예측    
# print("======================== 2. MCP 출력 ====================")

# path2 = 'C:\\프로그램\\ai5\\_save\\keras42\\'
# model = load_model(path2 + 'k42_02_0805_1412_0009-0.0000.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

y_pred = model.predict(x_test)
print(y_pred)
# [[0.25376353]
#  [0.25376353]
#  [0.25376353]
#  ...
#  [0.25376353]
#  [0.25376353]
#  [0.25376353]]

y_pred = np.round(y_pred)
print(y_pred)


print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print(acc)

print('acc : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)



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


