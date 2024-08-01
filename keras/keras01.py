import tensorflow as tf
print(tf.__version__)  #2.7.4

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()   # 순차적으로 연산하는 모델
model.add(Dense(1, input_dim=1)) # 인풋 한덩어리x, 아웃풋 한덩어리y.. 하이퍼 파라미터

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   # 컴퓨터가 알아먹게 컴파일한다. 최적화.. 하이퍼 파라미터
model.fit(x, y, epochs=1000)   # fit 훈련을 하겠다. epochs x와 y를 1000번 훈련시켜라.. 하이퍼 파라미터

#4. 평가, 예측
result = model.predict(np.array([4]))   # y가 없음. 미래의 y를 알기 위해 예측해 result에 저장
print("4의 예측값 : ", result)   # 최적의 가중치를 찾는게 목표
