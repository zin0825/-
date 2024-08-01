# https://dacon.io/competitions/open/236068/data
# 풀어라!!!


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터

path = "./_data/dacon/biabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)   # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)   # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(sample_submission_csv)   # [116 rows x 1 columns]

print(train_csv.shape, test_csv.shape, sample_submission_csv.shape)
# (652, 9) (116, 8) (116, 1)

print(train_csv.columns) # 9개 -1

print(train_csv.info())

print(train_csv.isna().sum())

# train_csv = train_csv.dropna()
# print(train_csv.isna().sum())

print(train_csv)   # [652 rows x 9 columns]

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


x = train_csv.drop(['Outcome'], axis=1)

print(x)   # [652 rows x 8 columns]

y = train_csv['Outcome']
print(y)

print(y.shape)   # (652,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=7575)

print(x.shape, y.shape)    # (652, 8) (652,)

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(64, activation='relu'))
model.add(Dense(148, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor= 'val_loss', mode='auto',
                   patience=20, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1680, batch_size=6220,
                 verbose=1, validation_batch_size=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 3))


y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
print("걸린시간 : ", round(end - start, 2), "초")

print(y_predict[:20])
y_predict = np.round(y_predict)
print(y_predict[:20])


accuracy_score = accuracy_score(y_test, y_predict)
print(y_predict)



y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_predict.shape)   # (116, 1)
y_submit = np.round(y_submit)
sample_submission_csv['Outcome'] = y_submit


sample_submission_csv.to_csv(path + "sample_sbmission_0723_1239.csv")

print("로스 : ", loss)




# 로스 :  0.27064642310142517
# ACC :  0.667

# 로스 :  0.2617517113685608
# ACC :  0.667
# 걸린시간 :  4.47 초

# 로스 :  [0.21972215175628662, 0.6818181872367859]

# 로스 :  [0.19491037726402283, 0.6818181872367859]

# 로스 :  [0.24453414976596832, 0.6818181872367859]

# 로스 :  [0.23192520439624786, 0.7121211886405945]

# 로스 :  [0.1500083953142166, 0.7575757503509521]

# 로스 :  [0.2740616202354431, 0.7121211886405945]

# 로스 :  [0.2821880877017975, 0.7121211886405945]

# 로스 :  [4.80870246887207, 0.6515151262283325]

# 로스 :  [2.4672627449035645, 0.5757575631141663]