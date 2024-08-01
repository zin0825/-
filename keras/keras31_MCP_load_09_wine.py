from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

from tensorflow.keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)

print(x.shape)
print(y.shape)
# (178, 13)
# (178,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_09_wine_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_09_wine_save.hdf5')  

loss2 = model2.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)

############################## 원래값 ###########################
# acc score :  0.9444444444444444
# 걸린 시간 :  1.92 초
# 로스 :  [0.08138372004032135, 0.9444444179534912]

# ======================== 1. save_model 출력 ====================
# 로스 :  [0.08138372004032135, 0.9444444179534912]
# r2 score :  0.9196777156060089
# ======================== 2. MCP 출력 ====================
# 로스 :  [0.08138372004032135, 0.9444444179534912]
# r2 score :  0.9196777156060089