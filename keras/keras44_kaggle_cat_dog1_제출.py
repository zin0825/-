# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview


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


import tensorflow as tf



#1. 데이터

path_train = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/'   
path_test = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path_mission = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

mission_csv = pd.read_csv(path_mission + 'sample_submission.csv', index_col=0)

start1 = time.time()


np_path = 'c:/프로그램/ai5/_data/_save_npy/'
# np.save(np_path + 'kares43_01_x_train.npy', arr=xy_train[0][0])   # xy_train을 여기에 저장
# np.save(np_path + 'kares43_01_y_train.npy', arr=xy_train[0][1])   # 통데이터로 저장해야 함
# np.save(np_path + 'kares43_01_x_test.npy', arr=xy_test[0][0])  
# np.save(np_path + 'kares43_01_y_test.npy', arr=xy_test[0][1])  




# x_train = np.load(np_path + 'kares43_01_x_train.npy')
# y_train = np.load(np_path + 'kares43_01_y_train.npy')
x_test = np.load(np_path + 'kares43_01_x_test.npy')
y_test = np.load(np_path + 'kares43_01_y_test.npy')


# print(x_train)
# print(x_train.shape)   # (20000, 100, 100, 3) 걸린 시간1 :  0.7 초
# print(y_train)
# print(y_train.shape)   # (20000,) 걸린 시간1 :  0.54 초
# print(x_test)
# print(x_test.shape)   # (100, 3) 걸린 시간1 :  0.55 초
# print(y_test)
# print(y_test.shape)   # (100, 3) 걸린 시간1 :  0.55 초

# x = x_train
# y = y_train

end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1,2), "초")



start = time.time()

# print(x_train.shape, y_test.shape)   # (17728, 200, 200, 3) (4432,)
# print(y_train.shape, y_test.shape)   # (17728,) (4432,)




# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(44, (2,2), input_shape=(100, 100, 3), activation='relu',
#                  strides=1, padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=44, kernel_size=(2,2), activation='relu', 
#                  strides=1, padding='same'))

# model.add(BatchNormalization())
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




# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10,
#                    verbose=1,
#                    restore_best_weights=True)


# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# import datetime
# date = datetime.datetime.now()
# # print(date)    
# # print(type(date))  
# date = date.strftime("%m%d_%H%M")
# # print(date)     
# # print(type(date))  

# path = './_save/keras42/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k42_02_', date, '_', filename])  


# ######################### cmp 세이브 파일명 만들기 끗 ###########################


# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath)


# start = time.time()
# hist = model.fit(x_train, y_train, epochs=160, batch_size=36,
#           verbose=1, 
#           validation_split=0.3,
#           callbacks=[es, mcp])

end = time.time()



#4. 평가, 예측    
print("======================== 2. MCP 출력 ====================")

path2 = 'C:\\프로그램\\ai5\\_save\\keras42\\'
model = load_model(path2 + 'k42_02_0805_1412_0009-0.0000.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)


print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)



#5. 파일 출력
y_submit = model.predict(x_test, batch_size=12)
mission_csv['label'] = y_submit

mission_csv.to_csv(path_mission + 'mission_0805_1424.csv')



# 걸린 시간1 :  0.33 초
# acc :  1.0
# 걸린 시간 :  0.0 초
# 로스 :  [3.3390790800224636e-23, 1.0]