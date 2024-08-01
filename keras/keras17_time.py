import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## 잘라라!!!
# train_test_split로 만 잘라라

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time   # 시간체크

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

#3. 컴파일, 훈련   # 한번 훈련시간 체크한다는 것은 어디서 체크??? -> 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1,   # 여기서부터 훈련시간
          verbose=1,
          validation_data=(x_val, y_val))
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([18])
print('로스 : ', loss)
print('[18]의 예측값 : ', results)
print('걸린시간 : ', round(end_time - start_time, 2), "초")

# 걸린시간 :  1.49 초


# 7/7 [==============================] - 0s 833us/step - loss: 0.0094 - val_loss: 0.0671
# 1/1 [==============================] - 0s 43ms/step - loss: 0.1984
# 로스 :  0.1983700841665268
# [11]의 예측값 :  [[10.730013]]

# 7/7 [==============================] - 0s 1ms/step - loss: 5.8186e-06 - val_loss: 4.5107e-05
# 1/1 [==============================] - 0s 34ms/step - loss: 1.2889e-04
# 로스 :  0.0001288938510697335
# [11]의 예측값 :  [[10.993002]]