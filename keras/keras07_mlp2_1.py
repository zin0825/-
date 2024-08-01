# [실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0],
]
)   #  예측 가능한 데이터

print(x.shape)

x = x.T
print(x.shape)


y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 에측
loss = model.evaluate(x,y)
result = model.predict([[10,1.3,0]])
print('로스 : ', loss)
print('[10,1.3,0]의 예측값 : ', result)



# (3, 10)
# (10, 3)
# 로스 :  0.004059500060975552
#[10,1.3,0]의 예측값 :  [[9.88149]]
