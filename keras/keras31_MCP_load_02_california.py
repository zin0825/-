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
from tensorflow.keras.models import Sequential, load_model
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


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_02_california_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_02_california_save.hdf5')
                    
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


############################## 원래값 ###########################

# 로스 :  0.3259637653827667
# r2스코어 :  0.7663240509145584
# 걸린시간 :  14.13 초

# ======================== 1. save_model 출력 ====================
# 로스 :  0.3259637653827667
# r2 score :  0.7663240509145584
# ======================== 2. MCP 출력 ====================
# 로스 :  0.3259637653827667
# r2스코어 :  0.7663240509145584