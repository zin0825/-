import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## 잘라라!!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


x_train = x [0:7]
y_train = y [0:7]

x_val = x [7:13]
y_val = y [7:13]

x_test = x [13:18]
y_test = y [13:18]

print(x_train)   # [1 2 3 4 5 6 7]
print(y_train)   # [1 2 3 4 5 6 7]
print(x_val)   # [ 8  9 10 11 12 13]
print(y_val)   # [ 8  9 10 11 12 13]
print(x_test)   # [14 15 16]
print(y_test)   # [14 15 16]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('로스 : ', loss)
print('[17]의 예측값 : ', results)

# 7/7 [==============================] - 0s 5ms/step - loss: 0.0232 - val_loss: 0.1596
# 1/1 [==============================] - 0s 62ms/step - loss: 0.4781
# 로스 :  0.47813376784324646
# [17]의 예측값 :  [[16.174313]]

# 7/7 [==============================] - 0s 4ms/step - loss: 0.0225 - val_loss: 0.1596
# 1/1 [==============================] - 0s 60ms/step - loss: 0.4736
# 로스 :  0.4735654890537262
# [17]의 예측값 :  [[16.179214]]