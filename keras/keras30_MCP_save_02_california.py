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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)

print(x)
print(y)
print(x.shape, y.shape)   # (20640, 8) (20640,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0000000000000004
print(np.max(x_test), np.max(x_test))   # 1.2491334943808423 1.2491334943808423




#2. 모델구성
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=8))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss',
                   mode= 'min',
                   patience=10,
                   verbose=1, 
                   restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras30_mcp/keras30_02_california.hdf5')


hist = model.fit(x_train, y_train, epochs=100, batch_size=64, 
          verbose=1, validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()


model.save('./_save/keras30_mcp/keras30_02_california_save.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

print("걸린시간 : ", round(end - start, 2), "초")


# 로스 :  0.31834354996681213
# r2스코어 :  0.7717867918008543
# 걸린시간 :  17.23 초