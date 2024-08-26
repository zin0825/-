
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau




#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


###### 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0

x_train = x_train.reshape(50000, 32*32*3)   # 3차원이라서 3이 붙음
x_test = x_test.reshape(10000, 32*32*3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=337)   # y는 예스 아니고 y


lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []


########## for 문 #############
for learning_rate in lr:

    #2. 모델
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))


    #3. 컴파일, 훈련
    es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                            patience=25, verbose=1,   # patience=10 10번 참아주겠다
                            factor=0.8,)   # facto 0.5씩 줄이겠다 / facto 만큼 곱해서 사용


    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))   
    # learning_rate=learning_rate 숫자 장난. 0.01로 넣어줘도 됨
    # learning_rate 디폴트 0.001

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          verbose=0,
          callbacks=[es, rlr],
          )


    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))




# same
# acc_score :  0.7224
# 걸린 시간 :  43.09 초
# 로스 :  [0.8071021437644958, 0.7224000096321106]


# acc_score :  0.471
# 걸린 시간 :  19.72 초
# 로스 :  [1.4911357164382935, 0.47099998593330383]


# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 2.317948341369629
# lr : 0.1, r2 : -0.0031692434963999537
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 2.3039188385009766
# lr : 0.01, r2 : -0.0003670886800281936
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 2.3028995990753174
# lr : 0.005, r2 : -0.00013119255962832366
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 1.951747179031372
# lr : 0.001, r2 : 0.08926260608961159
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 1.9017150402069092
# lr : 0.0005, r2 : 0.10491829505892325
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 1.970932960510254
# lr : 0.0001, r2 : 0.09012924487317531


# ReduceLROnPlateau
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 2.317948341369629
# lr : 0.1, r2 : -0.0031692378647640183
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 2.3039188385009766
# lr : 0.01, r2 : -0.0003670886254007799
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 2.3028995990753174
# lr : 0.005, r2 : -0.00013119255962832366
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 1.9567015171051025
# lr : 0.001, r2 : 0.09200455978470638
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 1.974894404411316
# lr : 0.0005, r2 : 0.08038539607046213
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 1.972902536392212
# lr : 0.0001, r2 : 0.08941583923496393