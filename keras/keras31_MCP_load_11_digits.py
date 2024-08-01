from sklearn.datasets import load_digits   # digits 숫자
import pandas as pd

x, y = load_digits(return_X_y=True)   # x와 y로 바로 반환해줌
print(x)
print(y)
print(x.shape, y.shape)   # (1797, 64) (1797,)   이미지는 0에서 225의 숫자를 부여함 225가 가장 진함 놈
# 1797장의 이미지가 있는데 8바이8 짜리를 64장으로 쭉 한것, 원래는 (1797,8,8)의 이미지 인데 칼라는 (1797,8,8,1)

print(pd.value_counts(y, sort=False))   # 확인, ascending=True 오름차순  # y라벨 10개
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (1797, 10)


y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (1797, 10)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (1797, 10)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)

print(x.shape, y.shape)   # (1797, 64) (1797, 10)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)   


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_11_digits_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_11_digits_save.hdf5')  


loss2 = model2.evaluate(x_test, y_test, verbose=0)    
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)

############################## 원래값 ###########################
# acc score :  0.9388888888888889
# 걸린 시간 :  6.83 초
# 로스 :  [0.17338207364082336, 0.9444444179534912]

#  ======================== 1. save_model 출력 ====================
# 로스 :  [0.17338207364082336, 0.9444444179534912]
# r2 score :  0.9144936715394134
# ======================== 2. MCP 출력 ====================
# 로스 :  [0.17338207364082336, 0.9444444179534912]
# r2 score :  0.9144936715394134

