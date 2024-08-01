import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(10))   # (0) = 함수 함수 함수
print(x)   # [[0 1 2 3 4 5 6 7 8 9]]  # 10개의 정수를 쏴주는거
print(x.shape)   # (10, )

x = np.array(range(1,11))
print(x)   # [1 2 3 4 5 6 7 8 9]
print(x.shape)   # (10,)

x = np.array([range(10), range(21,31), range(201,211)])   # 에러남 3개를 넣을 수 없음 -> []로 3개 묶어줌
# [ 1  2  3  4  5  6  7  8  9 10]
print(x)
print(x.shape)
# [[  0   1   2   3   4   5   6   7   8   9]
# [ 21  22  23  24  25  26  27  28  29  30]
# [201 202 203 204 205 206 207 208 209 210]]
# (3, 10)

x = x.T
print(x)
print(x.shape)
# [[  0  21 201]
# [  1  22 202]
# [  2  23 203]
# [  3  24 204]
# [  4  25 205]
# [  5  26 206]
# [  6  27 207]
# [  7  28 208]
# [  8  29 209]
# [  9  30 210]]
# (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])


# [실습]
# [10, 31, 211] 예측할것


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(1))

#3.커파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 에측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print('로스 : ', loss)
print('[10,31,211]의 예측값 : ', result)

# 로스 :  0.08351010829210281
# [10,31,211]의 예측값 :  [[10.400584]]

# 로스 :  0.6110557317733765
# [10,31,211]의 예측값 :  [[9.68859]]

# 로스 :  0.003180400701239705
# [10,31,211]의 예측값 :  [[10.928227]]
