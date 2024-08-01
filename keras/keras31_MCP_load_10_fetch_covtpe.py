from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (581012, 54) (581012,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))


# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (581012, 8)

y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (581012, 7)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)   # y는 예스 아니고 y

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# print(pd.value_counts(y_train))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_10_fetch_covtpe_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_10_fetch_covtpe_save.hdf5')  

loss2 = model2.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)

############################## 원래값 ###########################
# acc score :  0.7576159168359092
# 걸린 시간 :  5.82 초
# 로스 :  [0.5332769155502319, 0.7746893167495728]

# ======================== 1. save_model 출력 ====================
# 로스 :  [0.5332769155502319, 0.7746893167495728]
# r2 score :  0.44952499813432995
# ======================== 2. MCP 출력 ====================
# 로스 :  [0.5332769155502319, 0.7746893167495728]
# r2 score :  0.44952499813432995