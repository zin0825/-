from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 소수 네째자리까지 맞추면 합격. 예 : 6.0000 또는 5.9999

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))   # 인풋 딤 = 레이어의 갯수
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))


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

# 로스 :  0.38212570548057556
# 6의 예측값 :  [[5.6482525]]

# 로스 :  0.4475809931755066
# 6의 예측값 :  [[5.1059656]]

# 로스 :  0.3872317969799042
# 6의 예측값 :  [[5.5575566]]

# 'epochs :  1000
# 로스 :  0.5144163966178894
# 6의 예측값 :  [[4.8845606]]

# 'epochs :  1000
# 로스 :  0.38000017404556274
# 6의 예측값 :  [[5.699013]]

# 'epochs :  2000
# 로스 :  0.39595112204551697
# 6의 예측값 :  [[5.4860153]]

# 'epochs :  1700
# 로스 :  0.38000190258026123
# 6의 예측값 :  [[5.702233]]

# 'epochs :  1900
# 로스 :  0.37999996542930603
# 6의 예측값 :  [[5.700001]]

# 'epochs :  800
# 로스 :  0.38010281324386597
# 6의 예측값 :  [[5.6763535



# x = np.array([1,2,3,4,5])
# y = np.array([1,2,4,3,5])
# 'epochs :  1000
# 로스 :  0.3800000548362732
# 6의 예측값 :  [[5.7000046]]

# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
# 'epochs :  1000
# 로스 :  6.592473073396832e-05
# 6의 예측값 :  [[5.987187]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(5, input_dim=3))
# model.add(Dense(6, input_dim=5))
# model.add(Dense(4, input_dim=6))
# model.add(Dense(2, input_dim=4))
# 'epochs :  100
# 로스 :  0.04420331120491028
# 6의 예측값 :  [[5.3515162 5.8886166]]

# model = Sequential()
# model.add(Dense(7, input_dim=1))
# model.add(Dense(2, input_dim=3))
# model.add(Dense(4, input_dim=2))
# model.add(Dense(6, input_dim=4))
# model.add(Dense(2, input_dim=6))
#'epochs :  100
# 로스 :  0.327511191368103
# 6의 예측값 :  [[6.3406663 4.340892 ]]

# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(4, input_dim=4))
# model.add(Dense(3, input_dim=4))
# model.add(Dense(2, input_dim=3))
# model.add(Dense(1, input_dim=2))
# 'epochs :  100
# 로스 :  0.0005779670900665224
# 6의 예측값 :  [[5.966525]]

# model = Sequential()
# model.add(Dense(9, input_dim=1))
# model.add(Dense(4, input_dim=4))
# model.add(Dense(10, input_dim=4))
# model.add(Dense(5, input_dim=10))
# model.add(Dense(4, input_dim=5))
# 'epochs :  100
# 로스 :  0.045666880905628204
# 6의 예측값 :  [[5.9145856 5.9454327 5.7733183 5.774475 ]]

# model = Sequential()
# model.add(Dense(2, input_dim=1))
# model.add(Dense(7, input_dim=3))
# model.add(Dense(9, input_dim=7))
# model.add(Dense(6, input_dim=9))
# model.add(Dense(4, input_dim=6))
#'epochs :  100
# 로스 :  0.44559040665626526
# 6의 예측값 :  [[6.385632  5.968562  5.0616727 3.648164 ]]

# model = Sequential()
# model.add(Dense(4, input_dim=1))
# model.add(Dense(7, input_dim=8))
# model.add(Dense(3, input_dim=7))
# model.add(Dense(1, input_dim=3))
# 'epochs :  100
# 로스 :  0.0025963266380131245
# 6의 예측값 :  [[5.9080267]]

#'epochs :  100
# 로스 :  0.00020194209355395287
# 6의 예측값 :  [[6.0215645]]

# 'epochs :  100
# 로스 :  0.00426242733374238
# 6의 예측값 :  [[5.90685]]
