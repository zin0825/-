"""
01. 보스톤
02. california
03. diabetes
04. dacon_ddarung
05. kaggle_bike

06_cancer
07_dacon_diabetes
08_kaggle_bank
09_wine
10_fetch_covtpe
11_digits
"""


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


print(tf.__version__)   # 2.7.4


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

    


#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (20640, 8) (20640,)

x = x.reshape(20640, 8, 1, 1)   # 1, 1 내 맘대로

print(x.reshape, y.reshape)

x = x/255.

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)


#2. 모델구성
model = Sequential()
model.add(Conv2D(300, (2,1), activation='relu', input_shape=(8, 1, 1)))
model.add(Conv2D(300, (1,1), activation='relu'))
model.add(Conv2D(200, (1,1), activation='relu'))

model.add(Flatten())

model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

es = EarlyStopping(monitor = 'val_loss',
                   mode= 'min',
                   patience=10,
                   verbose=1, 
                   restore_best_weights=True)
 

hist = model.fit(x_train, y_train, epochs=10, batch_size=32, 
          verbose=1, validation_split=0.3,
          callbacks=[es])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

print("걸린시간 : ", round(end - start, 2), "초")



# 로스 :  0.5418245196342468
# r2스코어 :  0.6115782798620845
# 걸린시간 :  30.78 초
# 쥐피유 돈다!!!


# 로스 :  0.5869264602661133
# r2스코어 :  0.5792457153915529
# 걸린시간 :  18.28 초
