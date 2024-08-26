from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)


#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (581012, 54) (581012,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=337)   # y는 예스 아니고 y


lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []


########## for 문 #############
for learning_rate in lr:


    #2. 모델구성
    model = Sequential()
    # model.add(Dense(180, activation='relu', input_dim=x_train.shape[1]))
    # model.add(Dense(90))
    # model.add(Dense(46))
    # model.add(Dense(6))
    # model.add(Dense(7, activation='softmax'))

    model.add(Dense(10, activation='relu', input_dim=x_train.shape[1]))   # relu는 음수는 무조껀 0으로 만들어 준다.
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(1))

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))   
    # learning_rate=learning_rate 숫자 장난. 0.01로 넣어줘도 됨
    # learning_rate 디폴트 0.001

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=32, 
          )



    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))



# 케라스
# acc score :  0.48645485525455234
# 걸린 시간 :  8.08 초
# 로스 :  [0.93593430519104, 0.5576055645942688]

# acc score :  0.6604419813431551
# 걸린 시간 :  7.26 초
# 로스 :  [1.1363941431045532, 0.6739699244499207]

# ra- 3086, ep- 500, ba- 3086
# acc score :  0.8239991738666483
# 걸린 시간 :  483.44 초
# 로스 :  [0.401950865983963, 0.8311246037483215]

# ra- 3086, ep- 700, ba- 3086
# acc score :  0.8350142852225396
# 걸린 시간 :  675.04 초
# 로스 :  [0.37904050946235657, 0.8404185771942139]


# 사이킷런
# acc score :  0.7697497504388834
# 걸린 시간 :  61.56 초
# 로스 :  [0.5086667537689209, 0.7800419926643372]

# acc score :  0.7845168841003752
# 걸린 시간 :  57.34 초
# 로스 :  [0.47943201661109924, 0.7985267043113708]

# 스켈링
# acc score :  0.7627620391724897
# 걸린 시간 :  5.93 초
# 로스 :  [0.519395649433136, 0.777374267578125]

# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 1.9310747385025024
# lr : 0.1, r2 : -0.00378386198804348

# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 1.644215703010559
# lr : 0.01, r2 : 0.14532796030008843

# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 1.8912737369537354
# lr : 0.005, r2 : 0.016906603936606368

# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 2.280367136001587
# lr : 0.001, r2 : -0.18534803282455004

# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 2.485947847366333
# lr : 0.0005, r2 : -0.292209332650321

# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 2.094958543777466
# lr : 0.0001, r2 : -0.08897032865257648