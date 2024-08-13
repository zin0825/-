from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder



import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]



#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

from tensorflow.keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)   # (178, 3)

x = x.reshape(178, 13, 1, 1)

x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)



# #2. 모델구성
model = Sequential()
model.add(LSTM(180, input_shape=(13,1))) 
model.add(Dropout(0.4)) 
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.4)) 
model.add(Dense(46, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(22, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))


#2-2. 모델구성(함수형)
# input1 = Input(shape=(13,))
# dense1 = Dense(180, name='ys1')(input1)
# drop1 = Dropout(0.4)(dense1)
# dense2 = Dense(90, name='ys2')(drop1)
# drop2 = Dropout(0.4)(dense2)
# dense3 = Dense(46, name='ys3')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(22, name='ys4')(drop3)
# drop4 = Dropout(0.3)(dense4)
# dense5 = Dense(6, name='ys5')(drop4)
# output1 = Dense(3, activation='softmax')(dense5)
# model = Model(inputs=input1, outputs=output1)


# model = Sequential()
# model.add(Conv2D(180, (3,3), input_shape=(13,1,1), strides=1, activation='relu',padding='same')) 
# model.add(Dropout(0.4))
# model.add(Conv2D(90, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(Dropout(0.4))
# model.add(Conv2D(46, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Dropout(0.3))
# model.add(Flatten())                            

# model.add(Dense(22, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(units=3, activation='softmax'))


model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
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



path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k59_09_01_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################
 
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)
 
 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
print("걸린 시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


# acc score :  0.9444444444444444
# r2 score :  0.8387445818473455
# 걸린 시간 :  1.58 초
# 로스 :  [0.12170571833848953, 0.9444444179534912]
# 쥐피유 없다! xxxxx


# acc score :  0.9444444444444444
# r2 score :  0.8387445818473455
# 걸린 시간 :  2.82 초
# 로스 :  [0.09516128152608871, 0.9444444179534912]
# 쥐피유 돈다!!!


# dnn -> cnn
# acc score :  0.5
# r2 score :  -0.09573763428777693
# 걸린 시간 :  5.91 초
# 로스 :  [0.6602621674537659, 0.6111111044883728]

# acc score :  0.3888888888888889
# r2 score :  -0.2588412119223793
# 걸린 시간 :  4.96 초
# 로스 :  [0.7728832960128784, 0.7222222089767456]


# LSTM
# acc score :  0.5555555555555556
# r2 score :  -0.28035302571713655
# 걸린 시간 :  4.12 초
# 로스 :  [0.7705926299095154, 0.6111111044883728]
# k59_09_

# acc score :  0.6111111111111112
# r2 score :  -0.017815554435509712
# 걸린 시간 :  4.38 초
# 로스 :  [0.6722772121429443, 0.6666666865348816]
# k59_09_01_
