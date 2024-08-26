
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
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




#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_train.shape)   # (50000, 32, 32, 3) (50000, 32, 32, 3)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


#### 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

print(y_train.shape, y_test.shape)

### 원핫 y 1-1 케라스 //
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


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
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    # model.summary()



    #3. 컴파일,  훈련
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))   

    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=10, verbose=1,
                    restore_best_weights=True)

    model.fit(x_train, y_train,
            validation_split=0.2,
            epochs=18,
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



# acc_score :  0.4549
# 걸린 시간 :  44.36 초
# 로스 :  [2.0676424503326416, 0.45489999651908875]


# acc_score :  0.2046
# 걸린 시간 :  14.17 초
# 로스 :  [3.3856019973754883, 0.2046000063419342]


# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 4.6453328132629395
# lr : 0.1, r2 : -0.001002424982360648
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 4.609957695007324
# lr : 0.01, r2 : -0.00016506341480952867
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 4.608075141906738
# lr : 0.005, r2 : -0.00012576036810502011
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 4.6076788902282715
# lr : 0.001, r2 : -0.00011660716501803003
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 3.4437828063964844
# lr : 0.0005, r2 : 0.07796107648736986
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 3.5468437671661377
# lr : 0.0001, r2 : 0.06771873269153778