from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


# [실습] keras04 의 가장 좋은 레이어와 노드를 이용하여,
# 최소의 loss를 맹그러
# batch_size 조절
# 에포는 100으로 고정을 풀어주겠노라!!! 
# 로스 기준 0.32 미만!!!!

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))   # dim 디멘션 = 차원
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))

epochs = 100   # 데이터 숫자가 클 수록 전체 훈련 수 증가, 효율적.. 너무 많이 하면 과적합 발생
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=6)   # batch_size 단위로 잘라서 쓰겠다. 가중치 증가
# 몇 개의 관측치에 대한 예측을 하고, 레이블 값과 비교를 하는지를 설정하는 파라미터
# 배치사이즈가 100이면 전체 데이터에 대해 모두 예측한 뒤 실제 레이블 값과 비교한 후 가중치 갱신

#4. 평가, 예측
loss = model.evaluate(x,y)
print("============================")
print("'epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])   # [6] 벡터. 한덩어리
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




# -----------------
# model = Sequential()
# model.add(Dense(4, input_dim=1))
# model.add(Dense(7, input_dim=8))
# model.add(Dense(3, input_dim=7))
# model.add(Dense(1, input_dim=3))

# epochs = 100
#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=epochs, batch_size=6)
# 'epochs :  100
# 로스 :  0.3238098919391632
# 6의 예측값 :  [[5.858183]]

# model.fit(x, y, epochs=epochs, batch_size=12)
# 'epochs :  100
# 로스 :  0.3254922926425934
# 6의 예측값 :  [[5.894784]]

# model.fit(x, y, epochs=epochs, batch_size=12)
# 'epochs :  100
# 로스 :  0.3240296542644501
# 6의 예측값 :  [[5.8747416]]
# PS C:\프로그램\ai5> 

# model.fit(x, y, epochs=epochs, batch_size=11)
# 'epochs :  100
# 로스 :  0.32380959391593933
# 6의 예측값 :  [[5.8577943]]