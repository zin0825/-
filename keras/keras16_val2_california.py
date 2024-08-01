# keras11_2_cailfornia copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)


#2. 모델구성
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=8))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64, 
          verbose=0, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


# 로스 :  0.65132075548172
# r2스코어 :  0.5135331860964908

# relu
# 로스 :  0.5135491490364075
# r2스코어 :  0.631848257105937

# vervose
# 로스 :  0.46130672097206116
# r2스코어 :  0.6692997084031618
