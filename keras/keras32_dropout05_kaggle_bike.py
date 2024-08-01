import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import rc

#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\keggle\\bike-sharing-demand\\'   # 절대경로, 슬래시a, 슬래시t 특수문자로 경로 틀어짐
# path = 'C://프로그램//ai5//_data//bike-sharing-demand//'   # 절대경로
# path = 'C:/프로그램/ai5/_data/bike-sharing-demand/'   # 절대경로
# / // \ \\ 다 가능.
# 슬래시a, 슬래시b는 노란거 뜨는걸로 확인 특수문자로 인식

train_csv = pd.read_csv(path + "train.csv", index_col=0)  # pd는 판다스
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)   # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(sampleSubmission_csv.shape)   # (6493, 1)

print(train_csv.columns)
# 11개, 'casual', 'registered' 여기가 문제 두개 더하면 count임

print(train_csv.info())

print(test_csv.info())

print(train_csv.describe())   # 묘사


########### 결측치 확인 ############
print(train_csv.isna().sum())

print(train_csv.isnull().sum())

print(test_csv.isna().sum())

print(test_csv.isnull().sum())

########## x 와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# 봤더니 캐주얼과 레지스터가 있엇는데 그게 카운터였다
# [] 파이썬 리스트, 대괄호 묶어서 하나하나하나, 앞에는 인식하지만 뒤에는 인식 안됨
# 두 개 이상은 리스트, 한개짜리는 리스트 안해도 됨
print(x)   # [10886 rows x 8 columns] 

y = train_csv['count']
print(y)

print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=8150)

print(x)
print(y)
print(x.shape, y.shape)   # (10886, 8) (10886,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

test_csv = scaler.transform(test_csv)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0



#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))   # 한정시킬거야, relu 음수는 0 양수는 그래도
model.add(Dropout(0.3)) 
model.add(Dense(80, activation='relu'))   # 레이어는 깊은데 다 relu하고 싶을땐 
model.add(Dense(80, activation='relu'))   
model.add(Dropout(0.3)) 
model.add(Dense(80, activation='relu'))   
model.add(Dense(60, activation='relu'))   
model.add(Dropout(0.3)) 
model.add(Dense(60, activation='relu'))   
model.add(Dropout(0.3)) 
model.add(Dense(45, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(25, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))   # 위에서 다 처리했다 생각하고 마지막은 linear 

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(  
    monitor= 'val_loss',
    mode = 'min',   
    patience=10,
    verbose=1, 
    restore_best_weights=True) 

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   # 현재 시간
print(date)   # 2024-07-26 16:50:13.613311
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
print(type(date))



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_05',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)

hist = model.fit(x_train, y_train, epochs=1290, batch_size=764, 
                 verbose=1, validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()



#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   # (6493, 1)

sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
print(sampleSubmission_csv.shape)

sampleSubmission_csv.to_csv(path + "sampleSubmission_0729_1409.csv")

print('로스 : ', loss)
print('r2스코어 : ', r2)
print("걸리시간 : ", round(end - start, 2), "초")


# 로스 :  21045.46484375
# 걸리시간 :  6.66 초


# Drop
# 로스 :  37174.44921875
# r2스코어 :  -0.22766634058393498
# 걸리시간 :  2.39 초

