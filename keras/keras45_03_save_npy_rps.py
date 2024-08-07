# keras41_ImageDataGenerator5_rps


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,   # 수직 뒤집기
    # width_shift_range=0.1,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    # rotation_range=5,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    # fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문

path_train = './_data/image/rps/'   
# path_test = './_data/image/cat_and_dog/Test/' 

start1 = time.time()

xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(100,100),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=26000,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='categorical',   ## 다중분류 - 원핫 인코딩도 나와서 따로 할 필요 xx
    # class_mode='sparse',   ## 다중분류 
    # class_mode='binary'   # 이진분류
    # class_mode=None   # y값 없다!!!
    # color_mode='grayscale',   
    color_mode='rgb',
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

print(xy_train[0][0].shape)

np_path = 'c:/프로그램/ai5/_data/_save_npy/'
np.save(np_path + 'keras45_03_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_03_y_train.npy', arr=xy_train[0][1])   # 통데이터로 저장해야 함

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1],
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=44)

end1 = time.time()

print('걸린 시간 : ', round(end1 - start1, 2), "초")
# 걸린 시간 :  15.1 초

print(x_train.shape, y_train.shape)   # (2016, 100, 100, 3) (2016, 3)
print(x_test.shape, y_test.shape)   # (504, 100, 100, 3) (504, 3) 
