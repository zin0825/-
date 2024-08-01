import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
# 기준으로 위는 train 데이터 학습, 아래는 test 데이터 평가 
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)


# 7/7 [==============================] - 0s 332us/step - loss: 0.0153    # 훈련
# Epoch 100/100
# 7/7 [==============================] - 0s 332us/step - loss: 0.0144
# +++++++++++++++++++++++++++++++++++++++++
# 1/1 [==============================] - 0s 58ms/step - loss: 0.0592    # 평가
# 로스 :  0.059226710349321365
# [11]의 예측값 :  [[10.650603]]