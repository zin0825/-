import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\keggle\\bike-sharing-demand\\'   # 절대경로, 슬래시a, 슬래시t 특수문자로 경로 틀어짐
# path = 'C://프로그램//ai5//_data//bike-sharing-demand//'   # 절대경로
# path = 'C:/프로그램/ai5/_data/bike-sharing-demand/'   # 절대경로
# / // \ \\ 다 가능.
# 슬래시a, 슬래시b는 노란거 뜨는걸로 확인 특수문자로 인식

train_csv = pd.read_csv(path + "train.csv", index_col=0)   
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
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) 
print(x)    # [10886 rows x 8 columns]

y = train_csv['count']
print(y)

print(y.shape)   # (10886,)

x = x.to_numpy()

x = x.reshape(10886, 2, 2, 2)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=8150)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=200, kernel_size=2, input_shape=(10,1)))   # input_shape=(3,1) 행무시
model.add(Conv1D(200, 2))


model.add(Conv2D(180, (3,3), activation='relu', input_shape=(2,2,2), strides=1,padding='same')) 
model.add(Dropout(0.3)) 
model.add(Conv2D(filters=80, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(80, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Conv2D(80, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.3)) 
model.add(Conv2D(80, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Conv2D(60, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.3)) 
model.add(Conv2D(60, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.3)) 
model.add(Conv2D(45, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Conv2D(45, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.3)) 
model.add(Flatten())                            

model.add(Dense(units=45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(units=45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(units=45, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(units=25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='linear'))



#3. 컴파일, 훈련
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



path = './_save/keras39_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k39_05',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath)

hist = model.fit(x_train, y_train, epochs=1290, batch_size=764, 
                 verbose=1, validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print("걸린 시간 :", round(end-start,2),'초')
print('로스 : ', loss)

# 로스 :  22980.806640625
# r2스코어 :  0.24107112558755084
# 걸리시간 :  3.04 초
# 쥐피유 없다! xxxxx


# 로스 :  22973.14453125
# r2스코어 :  0.24132418042822312
# 걸리시간 :  3.96 초
# 쥐피유 돈다!!!


# dnn-> cnn
# r2스코어 :  0.15166143953556055
# 걸린 시간 : 6.25 초
# 로스 :  25688.181640625
