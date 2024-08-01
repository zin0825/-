# https://www.kaggle.com/competitions/playground-series-s4e1/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import rc


#1. 데이터
path = ".\\_data\\keggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(mission_csv.shape)    # (110023, 1)

print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

print(train_csv.isnull().sum())   # 결측치가 없다
print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

###############################################
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis=1)
print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
print(y.shape)                      # (165034,)

print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
print(pd.DataFrame(y).value_counts())
# 0         130113
# 1          34921

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=7680)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dropout(0.3)) 
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(148, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(168, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(10,))
dense1 = Dense(32, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(148, name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(168, name='ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(128, name='ys5')(drop4)
dense6 = Dense(64, name='ys6')(dense5)
dense7 = Dense(32, name='ys7')(dense6)
output1 = Dense(1, activation='sigmoid')(dense7)
model = Model(inputs=input1, outputs=output1)
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 10)]              0

#  ys1 (Dense)                 (None, 32)                352

#  dropout (Dropout)           (None, 32)                0

#  ys2 (Dense)                 (None, 64)                2112

#  dropout_1 (Dropout)         (None, 64)                0

#  ys3 (Dense)                 (None, 148)               9620

#  dropout_2 (Dropout)         (None, 148)               0

#  ys4 (Dense)                 (None, 168)               25032

#  dropout_3 (Dropout)         (None, 168)               0

#  ys5 (Dense)                 (None, 128)               21632

#  ys6 (Dense)                 (None, 64)                8256

#  ys7 (Dense)                 (None, 32)                2080

#  dense (Dense)               (None, 1)                 33

# =================================================================
# Total params: 69,117
# Trainable params: 69,117
# Non-trainable params: 0



# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# start_time = time.time()
# es = EarlyStopping(monitor='val_loss',   # arlyStopping 정의
#                    mode = 'min',               # 모르면 auto
#                    patience=10,
#                    restore_best_weights=True)

# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# import datetime   # 날짜
# date = datetime.datetime.now()   # 현재 시간
# print(date)   # 2024-07-26 16:50:13.613311
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
# print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
# print(type(date))



# path = './_save/keras32_mcp2/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# filepath = "".join([path, 'k32_08',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )

# hist = model.fit(x_train, y_train, epochs=2680, batch_size=6220, 
#                  validation_split=0.2, 
#                  callbacks=[es, mcp])
# end_time = time.time()


# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)
# print(y_predict[:20])       # y' 결과
# y_predict = np.round(y_predict)
# print(y_predict[:20])       # y' 반올림 결과

# accuracy_score = accuracy_score(y_test, y_predict)
# print("acc score : ", accuracy_score)
# print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# print("로스 : ", loss)


# y_submit = model.predict(test_csv)
# print(y_submit.shape)       # (110023, 1)

# y_submit = np.round(y_submit)
# mission_csv['Exited'] = y_submit
# mission_csv.to_csv(path + "sample_submission_0729_1407.csv")

'''
32 16 16 16 16 1
train_size=0.8, random_state=3434 / epochs=100, batch_size=16, validation_split=0.2
acc score :  0.7709923664122137

++++++++++++++++++++++++++++++
random_state=6666
random_state=1866
random_state=1186
'''

# random_state=7777
# 로스 :  0.3238998055458069

# random_state=7575
# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# 로스 :  0.32291173934936523

# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(168, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# epochs=1110
# 로스 :  0.3226291239261627

# epochs=1300

# random_state=7575
# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(148, activation='relu'))
# model.add(Dense(168, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# epochs=2680, batch_size=4220
# 로스 :  0.32392868399620056

# epochs=4680
# 로스 :  0.3231939375400543

# 로스 :  [0.32281380891799927, 0.8642106056213379]

# batch_size=6220
# 로스 :  [0.32281380891799927, 0.8642106056213379]

# random_state=7575
# acc score :  0.8646953676492866
# 걸린 시간 :  10.75 초
# 로스 :  [0.3234473764896393, 0.8646953701972961]




# Dropout
# acc score :  0.8603932499166843
# 걸린 시간 :  21.78 초
# 로스 :  [0.3330081105232239, 0.860393226146698]

# acc score :  0.8624534189717332
# 걸린 시간 :  28.97 초
# 로스 :  [0.3294872045516968, 0.8624534010887146]