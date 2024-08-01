import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_diabetes()
print(datasets)
print(datasets.DESCR)   # describe 확인 
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_03_diabetes_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_03_diabetes_save.hdf5')
                    
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

############################## 원래값 ###########################
# 로스 :  2368.745849609375
# r2스코어 :  0.5647267605952928

# ======================== 1. save_model 출력 ====================
# 로스 :  2368.745849609375
# r2 score :  0.5647267605952928
# ======================== 2. MCP 출력 ====================
# 로스 :  2368.745849609375
# r2스코어 :  0.5647267605952928