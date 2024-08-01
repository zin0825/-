from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 로스 기준 0.33 미만!!!!

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))   
model.add(Dense(10))   # 하이퍼파라미터 튜닝 - 현재 하고있는 레이어의 깊이와 노드 갯수 조절
model.add(Dense(30))
model.add(Dense(22))
model.add(Dense(4))
model.add(Dense(1))


epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("============================")
print("'epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)





# --------------------------------------

# model = Sequential()
# model.add(Dense(4, input_dim=1))
# model.add(Dense(7, input_dim=8))
# model.add(Dense(3, input_dim=7))
# model.add(Dense(1, input_dim=3))
# 'epochs :  100
# 로스 :  0.33883342146873474
# 6의 예측값 :  [[5.6892266]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(1))
# 'epochs :  100
# 로스 :  0.323955774307251

# model = Sequential()
# model.add(Dense(4, input_dim=1))
# model.add(Dense(6))
# model.add(Dense(4))
# model.add(Dense(6))
# model.add(Dense(3))
# model.add(Dense(1))
# 'epochs :  100
# 로스 :  0.32382732629776
# 6의 예측값 :  [[5.8608704]]

# 'epochs :  100
# 로스 :  0.32526543736457825
# 6의 예측값 :  [[5.8108106]]

# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model.add(Dense(4))
# model.add(Dense(6))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))
# 'epochs :  100
# 로스 :  0.3240807354450226
# 6의 예측값 :  [[5.880578]]

