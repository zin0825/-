


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

path_train = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/'   
path_test = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path_mission = 'C:/프로그램/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

mission_csv = pd.read_csv(path_mission + 'sample_submission.csv', index_col=0)



test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문


xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(100, 100),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=30000,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='rgb',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(100, 100),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=30000,   
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   


x = xy_train[0][0]
y = xy_train[0][1]


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



#4. 평가, 예측

print("======================== 2. MCP 출력 ====================")

path2 = './_data/_save/keras42/'
model = load_model(path2 + 'k42_02_0805_1412_0009-0.0000.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('로스 : ', loss)

y_submit = model.predict(xy_test, batch_size=12)
mission_csv['label'] = y_submit

mission_csv.to_csv(path_mission + 'teacher_0805_1419.csv')


