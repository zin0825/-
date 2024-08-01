import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt

#Hello 준영 I

#1. 데이터
path = './_data/dacon/따릉이/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

print(train_csv.shape)   # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission_csv.shape)   # (715, 1)

print(train_csv.columns)

print(train_csv.info())
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)   # [1328 rows x 10 columns]

print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)   # [1328 rows x 9 columns]

y = train_csv['count']
print(y)
print(y.shape)   # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)

print(x)
print(y)
print(x.shape, y.shape)   # (1328, 9) (1328,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.007490636704119841 1.0




#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=9))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(79, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
          verbose=1, validation_split=0.3)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0725_1545.csv")

print('로스 : ', loss)
print('걸린시간 : ', round(end - start, 2), "초")

# 로스 :  2208.22509765625
# 걸린시간 :  4.86 초

# 로스 :  2196.84375
# 걸린시간 :  5.29 초

# 스켈링
# 로스 :  1465.239501953125
# 걸린시간 :  4.88 초

# 로스 :  1479.632080078125
# 걸린시간 :  4.93 초

# StandardScaler 
# 로스 :  1876.7760009765625
# 걸린시간 :  4.99 초


