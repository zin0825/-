import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (442, 10) (442,)

x = x.reshape(442, 10, 1, 1)

x = x/255.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)



#2. 모델구성
model = Sequential()
model.add(Conv2D(300, (2,1), activation='relu', input_shape=(10, 1, 1)))
model.add(Conv2D(300, (1,1), activation='relu'))
model.add(Conv2D(200, (1,1), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu', input_dim=10))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

es = EarlyStopping(monitor = 'val_loss',
                   mode= 'min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)


hist = model.fit(x_train, y_train, epochs=100, batch_size=3, 
          verbose=1, validation_split=0.3,
          callbacks=[es])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

print("걸린시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


# 로스 :  2312.763916015625
# r2스코어 :  0.5750138480417163
# 걸린시간 :  2.54 초
# 쥐피유 없다! xxxxx

# 로스 :  2490.44970703125
# r2스코어 :  0.5423628463117105
# 걸린시간 :  7.8 초
# 쥐피유 돈다!!!

