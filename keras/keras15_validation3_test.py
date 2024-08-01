import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## 잘라라!!!
# train_test_split로 만 잘라라

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


"""
x_train = x [0:7]
y_train = y [0:7]

x_val = x [7:13]
y_val = y [7:13]

x_test = x [13:18]
y_test = y [13:18]
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.7, shuffle=False, random_state=0)



print(x_train)   # [1 2 3 4]
print(y_train)   # [1 2 3 4]
print(x_val)   # [ 5  6  7  8  9 10 11 12 13 14 15 16]
print(y_val)   # [ 5  6  7  8  9 10 11 12 13 14 15 16]
print(x_test)   # [13 14 15 16]
print(y_test)   # [13 14 15 16]


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


# 4/4 [==============================] - 0s 9ms/step - loss: 5.2209e-05 - val_loss: 0.0022
# 1/1 [==============================] - 0s 61ms/step - loss: 0.0044
# 로스 :  0.004376462660729885
# [17]의 예측값 :  [[17.080162]]

# 4/4 [==============================] - 0s 8ms/step - loss: 0.0286 - val_loss: 1.3892
# 1/1 [==============================] - 0s 60ms/step - loss: 2.7079
# 로스 :  2.7079033851623535
# [17]의 예측값 :  [[15.008954]]