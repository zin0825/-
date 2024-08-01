import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(10), range(21,31), (range(201,211))])

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]])
print(x.shape)   #(3, 10)
print(y.shape)   #(3, 10)

x = x.T
y = np.transpose(y)
print(x.shape)   #(10, 3)
print(y.shape)   #(10, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))   # x의 3개
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))   #여기가 y의 3개

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])

# np.set_printoptions(precision=3, suppress=True)
# precision 소수점 자리수까지 보기

print('로스 : ', loss)
print('[10,31,211]의 예측값 : ', result)



# 로스 :  7.94075870513916
# [10,31,211]의 예측값 :  [[5.618641  3.0293424 2.5633626]]

# 로스 :  1.5958588123321533
# [10,31,211]의 예측값 :  [[11.997391   1.0848652  2.5167964]]

