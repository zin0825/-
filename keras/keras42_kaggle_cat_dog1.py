# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview


'''
batsh_size=160
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
'''

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import natsort


import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"   
# fit 이 후 메모리 터질 때 사용하면 좀 더 나음

import tensorflow as tf


# file_path = "C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test/"
# file_names = natsort.natsorted(os.listdir(file_path))

# print(np.unique(file_names))
# i = 1
# for name in file_names:
#     src = os.path.join(file_path,name)
#     dst = str(i).zfill(5)+ '.jpg'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1



start1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=5,   # 정해진 각도만큼 이미지 회전
    zoom_range=1.5,   # 축소 또는 화대, 1.2배
    shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문

path_train = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/'   
path_test = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path_mission = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

mission_csv = pd.read_csv(path_mission + 'sample_submission.csv', index_col=0)



xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(80, 80),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=20000,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='rgb',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(80, 80),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=20000,   
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]   # 라벨이 몇개 인지 확인
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


# print(xy_train[0][0].shape)
# color_mode='grayscale',
# (30, 100, 100, 1)
# color_mode='rgb',
# (30, 100, 100, 3)

x = xy_train[0][0]
y = xy_train[0][1]

np_path = 'c:/프로그램/ai5/_data/_save_npy/'
np.save(np_path + 'keras42_01_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras42_01_01_y_train.npy', arr=xy_train[0][1])   # 통데이터로 저장해야 함
np.save(np_path + 'keras42_01_01_x_test.npy', arr=xy_test[0][0])  
np.save(np_path + 'keras42_01_01_y_test.npy', arr=xy_test[0][1])  

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=111)


xy_test = xy_test[0][0]


end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1,2), "초")

# # 걸린 시간 :  137.16 초

print(x_train.shape, y_test.shape)   # (17728, 200, 200, 3) (4432,)
print(y_train.shape, y_test.shape)   # (17728,) (4432,)




# #2. 모델 구성
# model = Sequential()
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
# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=60, batch_size=32,
#           verbose=1, 
#           validation_split=0.3,
#           callbacks=[es, mcp])
# end = time.time()



# #4. 평가, 예측      <- dropout 적용 X
# loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

# y_pred = model.predict(x_test)

# y_pred = np.round(y_pred)
# # print(y_pred)

# print('걸린 시간1 : ', round(end1 - start1,2), "초")

# acc = accuracy_score(y_test, y_pred)
# print('acc : ', acc)
# print('걸린 시간 : ', round(end - start,2), "초")
# print('로스 : ', loss)



# #5. 파일 출력
# y_submit = model.predict(xy_test, batch_size=12)
# mission_csv['label'] = y_submit

# mission_csv.to_csv(path_mission + 'mission_0805_1409.csv')



# # 걸린 시간1 :  91.05 초
# acc :  0.252
# 걸린 시간 :  68.28 초
# 로스 :  [0.5559946894645691, 0.25200000405311584]


# 걸린 시간1 :  135.26 초
# acc :  0.2514
# 걸린 시간 :  47.56 초
# 로스 :  [0.5616428256034851, 0.2513999938964844]


# 걸린 시간1 :  111.91 초
# acc :  0.2582
# 걸린 시간 :  77.77 초
# 로스 :  [0.5536549091339111, 0.2581999897956848]


# 걸린 시간1 :  102.07 초
# acc :  0.25521739130434784
# 걸린 시간 :  181.14 초


# 걸린 시간1 :  90.93 초
# acc :  0.25
# 걸린 시간 :  56.98 초
# 로스 :  [0.580438494682312, 0.25]


# 걸린 시간1 :  134.21 초
# acc :  0.25833333333333336
# 걸린 시간 :  234.68 초
# 로스 :  [0.5648844242095947, 0.25833332538604736]



# 걸린 시간1 :  90.86 초
# acc :  1.0
# 걸린 시간 :  195.82 초
# 로스 :  [0.0, 1.0]