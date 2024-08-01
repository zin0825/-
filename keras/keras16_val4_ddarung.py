import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
print(train_csv.info())

print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)   # [1328 rows x 9 columns]

y = train_csv['count']
print(y)

print(x.shape, y.shape)   # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)

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
model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          verbose=0, validation_split=0.3)

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

# submission_csv.to_csv(path + "submission_0719_0941.csv")



# 로스 :  1826.679931640625
# r2스코어 :  0.6341458244144845

# 로스 :  1639.746826171875
# r2스코어 :  0.7409352483709505

# 로스 :  1490.8359375
# r2스코어 :  0.764461806123707



