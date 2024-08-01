# 이름이 카멜케이스 (자바형식)
# 모델의 어떤 지점의 체크포인트


# keras26_Scaler01_boaton
# keras29_ModelCheckPoint5


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
from matplotlib import rc



#1. 데이터 
dataset = load_boston()
x = dataset.data   # x 데이터 분리   # 스켈링 할 것, x만 (비율만) 건들고 y는 건들면 안됨
y = dataset.target   # y 데이터 분리, sklearn 문법


# x_train과 x_test 하기전에 분리...를 여기보다

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=333)

# 여기서 하는게 더 좋음. 성능이 여기서 정상적으로 잘 작동하고 위는 70%정도로 나옴

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
# 13개의 데이터를 StandardScaler 로 스켈링 한다.
scaler = MinMaxScaler()

scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)   # 위 아래 두 줄을 하나로 줄일 수 있음
x_test = scaler.transform(x_test)   # 변환된 비율만 나오는 것 




#2. 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim=13))   # 특성은 많으면 좋음, 한계가 있음, 인풋딤에 다차원 행렬이 들어가면 안됨 
model.add(Dense(64, input_shape=(13,)))   # 이미지 input_shape=(8,8,1) ,하나 있는건 벡터이기 때문   # 13x10=140
model.add(Dropout(0.3))   # 30%를 제외하고 나머지를 훈련한다.  약 64개중에 45개만 훈련
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.3))   # 0.3 ,0.3, 0.3, 0.3으로 해도 상관없음
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))   # 과적합에 상당한 효과가 좋음, 피를 빼서 혈액순환을 해준것 
model.add(Dense(32, activation='relu'))   
model.add(Dropout(0.1))   
model.add(Dense(16, activation='relu'))   
model.add(Dense(1)) 



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
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
filepath = "".join([path, 'k32_01',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)   # 파일네임, 패스 더하면 요놈

start = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp],   # 두개 이상은 리스트
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')   # 두가지 다 저장할거야

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# print("걸린 시간 : ", round(end - start,2),'초')



# 로스 :  97.20042419433594
# r2 score :  0.008957412100013773
# 걸린 시간 :  1.84 초

# 스켈링
# 로스 :  23.58662223815918
# r2 score :  0.7595139881734306
# 걸린 시간 :  2.4 초



# 로스 :  20.869287490844727
# r2 score :  0.78721951962808
# 걸린 시간 :  5.2 초



# 로스 :  20.87381935119629
# r2 score :  0.7871733619432173


# dropout
# 로스 :  22.778947830200195
# r2 score :  0.767748922348566   # 0.02 성능 향상됨