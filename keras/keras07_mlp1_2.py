import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

# x = x.T
# x = x.transpose ()
x = np.transpose(x)   # 행과 열 바꿈


print(x.shape)   # (5,2)
print(y.shape)   # (5, )


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)   # 평가 모델의 로스값을 테스트 파일로 돌려서 손실값을 반환
results = model.predict([[6,11]])   # [[3,8]] 해도 됨 [n,2] 행 무시, 열 우선
print('로스 : ', loss)
print('[6, 11]의 예측값 : ', results)

# [실습] : 소수 2째자리까지 맞춰

# 기본
# results = model.predict([[6,11]])
# 로스 :  2.5497031153065564e-09
# [6, 11]의 예측값 :  [[5.9998894]]

# results = model.predict([[6,]])
# alueError: Exception encountered when calling layer "sequential" (type Sequential).


# 로스 :  0.00034689524909481406
# [6, 11]의 예측값 :  [[5.9598193]]
