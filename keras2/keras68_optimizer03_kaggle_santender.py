# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

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
tf.random.set_seed(2038)
np.random.seed(2038)
rn.seed(2038)



#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)   # (200000, 201)
# print(test_csv.shape)   # (200000, 200)
# print(sample_csv.shape)   # (200000, 1)

# print(train_csv.columns)

x = train_csv.drop(['target'], axis=1)
# print(x)   # [200000 rows x 200 columns]

y = train_csv['target']
# print(y)

print(x.shape, y.shape)   # (200000, 200) (200000,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038,
                                                    stratify=y)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (200000, 200) (200000,)


lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

########## for 문 #############

for learning_rate in lr:


    #2. 모델구성
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(400))
    model.add(Dense(340))
    model.add(Dense(330))
    model.add(Dense(250))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1, activation='sigmoid'))


    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))   

    model.fit(x_train, y_train,
            validation_split=0.2,
            verbose=0,
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



# acc score :  0.87495
# 걸린 시간 :  104.54 초
# 로스 :  [1.0752360820770264, 0.8749499917030334]

# batch_size=1000
# acc score :  0.897375
# 걸린 시간 :  117.65 초
# 로스 :  [0.3148635923862457, 0.8973749876022339]

# batch_size=5000
# acc score :  0.9049
# 걸린 시간 :  58.69 초
# 로스 :  [0.25515782833099365, 0.9049000144004822]

# acc score :  0.91195
# 걸린 시간 :  60.1 초
# 로스 :  [0.24282044172286987, 0.9119499921798706]

# acc score :  0.91185
# 걸린 시간 :  58.62 초
# 로스 :  [0.24183888733386993, 0.9118499755859375]

# 스켈링
# acc score :  0.912525
# 걸린 시간 :  120.47 초
# 로스 :  [0.2372458279132843, 0.9125249981880188]

# learning_rate
# =================1. 기본 출력 =================
# lr : 0.1, 로스 : 29314222080.0
# lr : 0.1, r2 : -0.11172873818788198
# =================1. 기본 출력 =================
# lr : 0.01, 로스 : 0.33511289954185486
# lr : 0.01, r2 : -0.022279853175895648
# =================1. 기본 출력 =================
# lr : 0.005, 로스 : 0.32703059911727905
# lr : 0.005, r2 : -0.0015796640732366196
# =================1. 기본 출력 =================
# lr : 0.001, 로스 : 0.274767130613327
# lr : 0.001, r2 : 0.15143587383201051
# =================1. 기본 출력 =================
# lr : 0.0005, 로스 : 0.2582469880580902
# lr : 0.0005, r2 : 0.18718510309286318
# =================1. 기본 출력 =================
# lr : 0.0001, 로스 : 0.27203744649887085
# lr : 0.0001, r2 : 0.1315928063915902


