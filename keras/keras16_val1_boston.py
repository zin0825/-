# keras11_1_boston copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # Description 속성을 이용해서 데이터셋의 정보를 확인
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=33)

print(x)
print(y)
print(x.shape, y.shape)   # (506, 13) (506,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=13))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation= 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64, 
          verbose=0, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


# 로스 :  70.41497802734375
# r2스코어 :  0.16564466530069877

# relu
# train_size=0.8,
# shuffle=True,
# random_state=150
# model.add(Dense(20, activation='relu', input_dim=13))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation= 'linear'))
# # 로스 :  39.54791259765625
# r2스코어 :  0.5313922968728959

# 로스 :  16.31851577758789
# r2스코어 :  0.6445978443931606

# 로스 :  15.1525297164917
# r2스코어 :  0.669991942085683