# keras18_overfit1_boston


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
from matplotlib import rc



#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # describe 확인 
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data   # x 데이터 분리
y = dataset.target   # y 데이터 분리, sklearn 문법

# print(x)

# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=333)
print(x.shape)   # (506, 13)
print(y.shape)   # (506, )

#2. 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim=13))   # 특성은 많으면 좋음, 한계가 있음, 인풋딤에 다차원 행렬이 들어가면 안됨 
model.add(Dense(32, input_shape=(13,)))   # 이미지 input_shape=(8,8,1) ,하나 있는건 벡터이기 때문 13,
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1, 
          validation_split=0.1
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

print("걸린 시간 : ", round(end - start,2),'초')

print("=================== hist ==================")
print(hist)

print("================ hist.history =============")
print(hist.history)

print("================ loss =============")
print(hist.history['loss'])
print("================ val_loss =============")
print(hist.history['val_loss'])
print("==================================================")




# 로스 :  97.20042419433594
# r2 score :  0.008957412100013773
# 걸린 시간 :  1.84 초
