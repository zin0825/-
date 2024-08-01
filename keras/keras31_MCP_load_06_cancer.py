import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_breast_cancer()
# print(datasets)   # 한개의 행[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01, 1.189e-01]
#  y 값 array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...

print(datasets.DESCR)   
# Number of Instances: 569 행
# Attributes: 30  속성, 열
# Missing Attribute Values: None 결측치 있냐
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (569, 30) (569,)

# y는 넘파이 데이터에서 암이 반절이 걸렸다. 0,1,2, 다중 분류일 경우 다 확인해야하는데
# y의 라벨값을 묻는것이 있다.
# 넘파이에서 y가 0과 1의 종류

print(type(x))   #<class 'numpy.ndarray'>

"""np.unique(datasets)
unique, counts= np.unique(datasets, return_counts=True)
print(unique, counts)
dict(Zip(unique, counts))
"""

# 0과 1의 갯수가 몇개인지 찾아요.
print(np.unique(y, return_counts=True))   
# (array([0, 1]), array([212, 357], dtype=int64))

# print(y.value_counts())   에러
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=99, 
                                                    train_size=0.9)

print(x_train.shape, y_train.shape)   # (398, 30) (398,)
print(x_test.shape, y_test.shape)   # (171, 30) (171,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_06_cancer_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_06_cancer_save.hdf5')  

loss2 = model2.evaluate(x_test, y_test, verbose=0)    
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)


############################## 원래값 ###########################
# acc_score :  0.9649122807017544
# 걸리시간 :  3.25 초
# 로스 :  0.025403909385204315

#  ======================== 1. save_model 출력 ====================
# 로스 :  [0.025403909385204315, 0.9649122953414917]
# r2 score :  0.8786216171722117
# ======================== 2. MCP 출력 ====================
# 로스 :  [0.025403909385204315, 0.9649122953414917]
# r2 score :  0.8786216171722117
