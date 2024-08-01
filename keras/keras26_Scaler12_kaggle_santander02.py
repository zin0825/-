# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
path = "C:\\프로그램\\ai5\\_data\\keggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)   # (200000, 201)
# print(test_csv.shape)   # (200000, 200)
# print(sample_csv.shape)   # (200000, 1)

# print(train_csv.columns)

x = train_csv.drop(['target'], axis=1)
# print(x)   # [200000 rows x 200 columns]

y = train_csv['target']
# print(y)
# print(y.shape)   # (200000,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038,
                                                    stratify=y)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (200000, 200) (200000,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

test_csv = scaler.transform(test_csv)

print(x_train)   
print(np.min(x_train), np.max(x_train))   # .0 1.0000000000000002
print(np.min(x_test), np.max(x_test))   # -0.07986086462696101 1.044658899717469


#2. 모델구성
model = Sequential()
model.add(Dense(400, activation='relu', input_dim=200))
model.add(Dense(400))
model.add(Dense(340))
model.add(Dense(340))
model.add(Dense(140))
model.add(Dense(130))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=50, batch_size=3400, 
                 validation_batch_size=0.3,
                 callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)



y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
sample_csv['target'] = y_submit
sample_csv.to_csv(path + "sample_submission_0724_1711.csv")

accuracy_score = accuracy_score(y_test, y_pred)

print('acc score : ', accuracy_score)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)


# acc score :  0.87495
# 걸린 시간 :  104.54 초
# 로스 :  [1.0752360820770264, 0.8749499917030334]

# batch_size=1000
# acc score :  0.897375
# 걸린 시간 :  117.65 초
# 로스 :  [0.3148635923862457, 0.8973749876022339]

# batch_size=5000
# acc score :  0.9049
# 걸린 시간 :  58.69 초
# 로스 :  [0.25515782833099365, 0.9049000144004822]

# acc score :  0.91195
# 걸린 시간 :  60.1 초
# 로스 :  [0.24282044172286987, 0.9119499921798706]

# acc score :  0.91185
# 걸린 시간 :  58.62 초
# 로스 :  [0.24183888733386993, 0.9118499755859375]

# 스켈링
# acc score :  0.912525
# 걸린 시간 :  120.47 초
# 로스 :  [0.2372458279132843, 0.9125249981880188]

# StandardScaler
# acc score :  0.8798
# 걸린 시간 :  25.87 초
# 로스 :  [0.5471627712249756, 0.879800021648407]

# acc score :  0.880575
# 걸린 시간 :  64.97 초
# 로스 :  [1.0169141292572021, 0.8805750012397766]