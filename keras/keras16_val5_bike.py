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
sampleSubmossion_csv = pd.read_csv(path + "samplesubmission.csv", index_col=0)

print(train_csv.shape)   # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(sampleSubmossion_csv.shape)   # (6493, 1)


print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

print(train_csv.info())

print(test_csv.info())

print(train_csv.describe())

######### 결측치 확인 #########
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)   # [10886 rows x 8 columns]

y = train_csv['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=150)
print(x)
print(y)
print(x.shape, y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64, 
          verbose=0, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

sampleSubmossion_csv['count'] = y_submit
print(sampleSubmossion_csv)
print(sampleSubmossion_csv.shape)

# sampleSubmossion_csv.to_csv(path + "samplSubmission_0719_0933.csv")

print('로스 : ', loss)


# 로스 :  23063.072265625

# 로스 :  22537.154296875 

