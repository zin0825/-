
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
tf.random.set_seed(2038)
np.random.seed(2038)
rn.seed(2038)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)   # 샘플만 요약해 나와서 000만 나옴
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  6만의 28, 28, 1 / 컬러는 6만의 28, 28, 3
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)   만의 28, 28, 1

# x는 6만의 28, 28, 1로 리쉐이프
# y는 원핫앤코딩


# x_train = x_train.reshape(60000, 28, 28, 1)   # 1이 없으면 2차원이니까 3차원으로 정의하기 위해
# x_test = x_test.reshape(10000, 28, 28, 1)



# # y_train = pd.get_dummies(y_train)   # 이걸 해야 10이 생김
# # y_test = pd.get_dummies(y_test)
# # print(y_train)   # [60000 rows x 10 columns]
# # print(y_test)   # [10000 rows x 10 columns]

# # print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
# # print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)


# ##### 스켈일링 1-1   # 이 데이터는 이 안에 넣어줘야겠다
x_train = x_train/255.   # 소수점
x_test = x_test/255.

# print(np.max(x_train), np.min(x_train))   # 1.0 0.0


x_train = x_train.reshape(60000, 28 * 28)   # 흑백 1은 묵음
x_test = x_test.reshape(10000, 28 * 28)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038)
# # # 여기까지가 #1 완성 --------------------

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

########## for 문 #############

for learning_rate in lr:
    
        
    #2. 모델
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1]))   # 흑백 1은 묵음
    # 맹그러봐
    model.add(Dense(128, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(44, activation='relu'))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(10))

    # model.summary()

    #3 컴파일, 훈련
    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=30, verbose=1,
                    restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                            patience=25, verbose=1,   # patience=10 10번 참아주겠다
                            factor=0.8,)   # facto 0.5씩 줄이겠다 / facto 만큼 곱해서 사용


    
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))   

    model.fit(x_train, y_train,
              validation_split=0.2,
              verbose=0,
              epochs=1,
              callbacks=[es, rlr],
              )


    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

# acc_score :  0.9758
# 걸린 시간 :  18.6 초
# 로스 :  [0.10267635434865952, 0.9757999777793884]


# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 0.09002415835857391
# lr : 0.1, r2 : -0.0007032230085884273
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 0.09045851975679398
# lr : 0.01, r2 : -0.005466588529789029
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 0.014705879613757133
# lr : 0.005, r2 : 0.8339128645683814
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 0.010346507653594017
# lr : 0.001, r2 : 0.8834513378844304
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 0.009667912498116493
# lr : 0.0005, r2 : 0.891260600057619
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 0.01871619187295437
# lr : 0.0001, r2 : 0.7890899959550425


# ReduceLROnPlateau
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 0.09001760184764862
# lr : 0.1, r2 : -0.0006292447472065366
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 0.09045851975679398
# lr : 0.01, r2 : -0.005466575128490647
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 0.015489097684621811
# lr : 0.005, r2 : 0.8250669208956232
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 0.009437073022127151
# lr : 0.001, r2 : 0.8937204514767437
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 0.009634171612560749
# lr : 0.0005, r2 : 0.8915267205886472
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 0.018645232543349266
# lr : 0.0001, r2 : 0.7898832551797239


