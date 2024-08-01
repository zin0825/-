from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)   # y는 예스 아니고 y

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# print(pd.value_counts(y_train))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0



# #2. 모델구성
# model = Sequential()
# model.add(Dense(180, activation='relu', input_dim=54))
# model.add(Dropout(0.3))
# model.add(Dense(90))
# model.add(Dropout(0.3))
# model.add(Dense(46))
# model.add(Dropout(0.3))
# model.add(Dense(6))
# model.add(Dense(7, activation='softmax'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(54,))
dense1 = Dense(180, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(90, name='ys2')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(46, name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(6, name='ys4')(drop3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 54)]              0

#  ys1 (Dense)                 (None, 180)               9900

#  dropout (Dropout)           (None, 180)               0

#  ys2 (Dense)                 (None, 90)                16290

#  dropout_1 (Dropout)         (None, 90)                0

#  ys3 (Dense)                 (None, 46)                4186

#  dropout_2 (Dropout)         (None, 46)                0

#  ys4 (Dense)                 (None, 6)                 282

#  dense (Dense)               (None, 7)                 49

# =================================================================
# Total params: 30,707
# Trainable params: 30,707
# Non-trainable params: 0



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



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_10',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
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



if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")


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