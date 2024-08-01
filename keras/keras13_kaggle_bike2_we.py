# 기존 캐글 데이터에서
# 1. train_csv의  y를 casual과 register로 잡는다.
#    그래서 훈련을 해서 test_csv의 casual과 register를 predict한다.
   
# 2. test_csv에 casual과 register 컬럼을 합쳐

# 3. train_csv에 y를 count로 잡는다.

# 4. 전체 훈련

# 5. test_csv 예측해서  submission 에 붙여!!!


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\keggle\\bike-sharing-demand\\'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test_csv.shape)
# print(sampleSubmission_csv.shape)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]

print(x.info())
print(y.columns)   # Index(['casual', 'registered']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    random_state=100)

# print('x_train : ', x_train.shape)
# print('x_test : ', x_test.sahpe)
# print('y_train : ', y_train.shape)
# print('y_test : ', y_test.shape) 


#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))
model.add(Dense(95, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

y_submit = model.predict(test_csv)
casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]
print(casual_predict)
test_csv = test_csv.assign(casual = casual_predict, registered = registered_predict)
test_csv.to_csv(path + "test_bike2.csv")

print('로스 : ', loss)

# 로스 :  8651.05859375

# 로스 :  8645.19921875
# r2스코어 :  0.4227356641076878
# [5.190613  4.5502768 4.5502768 ... 3.6593547 5.0882344 2.7138748]






