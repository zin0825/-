from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]



#1. 데이터
datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (581012, 54) (581012,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))


# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (581012, 8)

y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (581012, 7)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (581012, 7)


x = x.reshape(581012, 54, 1)

x = x/255.



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)   # y는 예스 아니고 y


# print(pd.value_counts(y_train))



# #2. 모델구성
model = Sequential()
model.add(LSTM(180, input_shape=(54, 1))) 
model.add(Dropout(0.3))
model.add(Dense(90))
model.add(Dropout(0.3))
model.add(Dense(46))
model.add(Dropout(0.3))
model.add(Dense(6))
model.add(Dense(7, activation='softmax'))


# #2-2. 모델구성(함수형)
# input1 = Input(shape=(54,))
# dense1 = Dense(180, name='ys1')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(90, name='ys2')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(46, name='ys3')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(6, name='ys4')(drop3)
# output1 = Dense(7, activation='softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


# model = Sequential()
# model.add(Conv2D(180, (3,3), input_shape=(54,1,1), strides=1, activation='relu',padding='same')) 
# model.add(Dropout(0.3))
# model.add(Conv2D(90, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(Dropout(0.3))
# model.add(Conv2D(46, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Dropout(0.3))
# model.add(Flatten())                            

# model.add(Dense(6, activation='relu'))
# model.add(Dense(units=7, activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
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
filepath = "".join([path, 'k59_10_01_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
) 

hist = model.fit(x_train, y_train, epochs=10, batch_size=2586, 
                 validation_split=0.3,
                 callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('ACC : ', round(loss[1], 3))

y_pred = model.predict(x_test)
print(y_pred[:20])
y_pred = np.round(y_pred)
print(y_pred[:20])

accuracy_score = accuracy_score(y_test,np.round(y_pred))
print(y_pred)

print('acc score : ', accuracy_score)

r2 = r2_score(y_test, y_pred)
print('r2 score : ', r2)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)





# acc score :  0.7553096278957695
# r2 score :  0.2650155504329697
# 걸린 시간 :  9.48 초
# 로스 :  [0.5386325716972351, 0.7712643146514893]


# acc score :  0.6992186155381914
# r2 score :  0.12248471311788793
# 걸린 시간 :  9.39 초
# 로스 :  [0.646875262260437, 0.7189942002296448]
# 쥐피유 없다! xxxxx


# acc score :  0.6967230043716223
# r2 score :  0.12221774066302624
# 걸린 시간 :  6.5 초
# 로스 :  [0.6494757533073425, 0.7203366756439209]
# 쥐피유 돈다!!!


# dnn -> cnn
# acc score :  0.6845031152111803
# r2 score :  0.05502278038815386
# 걸린 시간 :  55.18 초
# 로스 :  [0.6950746774673462, 0.7026952505111694]


# LSTM
# acc score :  0.6776014595022547
# r2 score :  0.03582574876188259
# 걸린 시간 :  45.44 초
# 로스 :  [0.7117458581924438, 0.6933495998382568]
# k59_10_

# acc score :  0.6772228150493959
# r2 score :  0.023929026784396075
# 걸린 시간 :  45.26 초
# 로스 :  [0.7346964478492737, 0.6880830526351929]
# k59_10_01_