# keras16_val2_california.py



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (2040, 8) (20640, )

# [실습] 만들기
# R2 0.59 이상

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)

print(x)
print(y)
print(x.shape, y.shape)   # (20640, 8) (20640,)

#2. 모델구성
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=8))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, 
          verbose=1, validation_split=0.2)
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

print("걸린시간 : ", round(end - start, 2), "초")

print("================= hist ====================")
print(hist)
print("================== hist.history =============")
print(hist.history)
print("=================== loss ==================")
print(hist.history['loss'])
print("================= val_loss =================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'  

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()



# 로스 :  0.489817351102829
# r2스코어 :  0.6488610970964998
# 걸린시간 :  10.57 초

# 로스 :  0.5203878879547119
# r2스코어 :  0.6269457920865135
# 걸린시간 :  10.76 초