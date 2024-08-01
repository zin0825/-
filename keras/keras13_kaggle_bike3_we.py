import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1.  데이터
path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test_bike2.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test_csv.shape)

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x.shape)   # (10886, 10)
print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=10)

#2. 모델구성
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=10))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))

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
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "sampleSubmission_col_0718_1245.csv")


# random_state=100
# 로스 :  0.04266536608338356
# r2스코어 :  0.9999986325081562

# random_state=110
# 로스 :  0.01330199372023344
# r2스코어 :  0.9999995943391201

# random_state=10
# 로스 :  0.01036435179412365
# r2스코어 :  0.9999997025847606

# model.add(Dense(20, activation='relu', input_dim=10))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(45, activation='relu'))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(1, activation='linear'))
# 로스 :  0.012512844987213612
# r2스코어 :  0.9999996409334256
