# CNN으로 맹그러

"""
01. 보스톤
02. california
03. diabetes
04. dacon_ddarung
05. kaggle_bike

06_cancer
07_dacon_diabetes
08_kaggle_bank
09_wine
10_fetch_covtpe
11_digits
"""


import sklearn as sk

from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




#1. 데이터 #1. 데이터 
dataset = load_boston()

x = dataset.data   
y = dataset.target  

print(x.shape)      # (506, 13)    # 무슨 값이 알고싶어

# x = x.reshape(506,13,1,1)   # 정의된 값을 변환할거야
# y = y.reshape(506,1,1,1)

x = x/225.


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)   # (404, 13) (102, 13)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(90, (2,1), input_shape=(13,1,1), activation='relu', strides=1, padding='same'))
model.add(Conv2D(filters=90, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(90, (2,1), activation='relu', strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(66, activation='relu'))
model.add(Dense(66, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
                   restore_best_weights=True)

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   
print(date)  
print(type(date))  
date = date.strftime("%m%d_%H%M")   
print(date)   
print(type(date))



path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   


filepath = "".join([path, 'k39_01',date, '_' , filename]) 


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath)    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# 파일네임, 패스 더하면 요놈

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=16,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1)

print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2_score : ', r2)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)


# r2_score :  0.7912355514442061
# 걸린 시간 :  15.51 초
# 로스 :  [15.604340553283691, 0.0]


# dnn -> cnn
# r2_score :  0.8929747767934547
# 걸린 시간 :  12.91 초
# 로스 :  [7.9997239112854, 0.0]

# r2_score :  0.8474114652075396
# 걸린 시간 :  7.24 초
# 로스 :  [11.40540599822998, 0.0]