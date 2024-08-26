import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sklearn as sk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)


#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=5,   # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,   # 축소 또는 화대, 1.2배
    shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)


start1 = time.time()


np_path = 'C:\\ai5\\_data\\_save_npy\\'

x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,  # test 안 할 때 주의!!
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=44)



print(x_train)
print(x_train.shape)   # 
print(y_train)
print(y_train.shape)   #


# augment_size =  10000

# print(x_train.shape[0]) 

# randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)
# print(x_train[0].shape)


# x_augmented = x_train[randidx].copy()
# y_augmented = y_train[randidx].copy()
# print(x_augmented.shape, y_augmented.shape)


# x_augmented = train_datagen.flow(x_augmented, y_augmented,
#                                  batch_size=augment_size,
#                                  shuffle=False).next()[0]

# print(x_augmented.shape) 

# print(x_train.shape, x_test.shape)


# x_train = np.concatenate((x_train, x_augmented)) 
# y_train = np.concatenate((y_train, y_augmented))


# x_train = x_train.reshape(
#     x_train.shape[0],
#     x_train.shape[1],
#     x_train.shape[2]*x_train.shape[3])

# x_test = x_test.reshape(
#     x_test.shape[0],
#     x_test.shape[1],
#     x_test.shape[2]*x_test.shape[3])


# print(x_train.shape, y_train.shape)   # (12016, 100, 300) (12016, 3)


# end1 = time.time()

# print('걸린 시간1 : ', round(end1 - start1, 2), "초")


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=337)   # y는 예스 아니고 y


lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []


########## for 문 #############
for learning_rate in lr:


    #2. 모델 구성
    model = Sequential()
    # model.add(Conv2D(110, (2,2), input_shape=(100, 100, 3), activation='relu',
    #                  strides=1, padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(filters=110, kernel_size=(2,2), activation='relu', 
    #                  strides=1, padding='same'))
    # model.add(Conv2D(110, (2,2), strides=1, padding='same'))
    # model.add(Flatten())

    model.add(Dense(110, activation='reul', input_dim=x_train.shape[1]))
    model.add(Dense(110, activation='relu',))
    model.add(Dense(100, activation='relu',))
    model.add(Dense(80, activation='relu',))
    model.add(Dense(20, activation='relu',))
    model.add(Dense(3, activation='softmax'))


    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))   


    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          verbose=0
          )


    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))



# acc :  0.996031746031746
# 로스 :  [0.019349033012986183, 0.9960317611694336]


# acc :  1.0
# r2_score :  1.0
# 걸린 시간 :  89.5 초
# 로스 :  [0.000641791382804513, 1.0]

# acc :  1.0
# r2_score :  1.0
# 걸린 시간 :  86.22 초
# 로스 :  [0.00030748904100619256, 1.0]