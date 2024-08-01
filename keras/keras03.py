from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np   # 수치와 연산

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# [실습] 맹그러봐!!

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))   # y = ax + b (함수)

epochs = 1000
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)   # 위에서 설정한 epochs 값

#4. 평가, 예측
loss = model.evaluate(x,y)   # 평가하는거
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
# 6의 예측값 :  [[5.6763535]]

