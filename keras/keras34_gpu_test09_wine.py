from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)

print(x.shape)
print(y.shape)
# (178, 13)
# (178,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.05128205128205132 0.9896551724137932





# #2. 모델구성
# model = Sequential()
# model.add(Dense(180, activation='relu', input_dim=13))
# model.add(Dropout(0.4)) 
# model.add(Dense(90, activation='relu'))
# model.add(Dropout(0.4)) 
# model.add(Dense(46, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(22, activation='relu'))
# model.add(Dropout(0.3)) 
# model.add(Dense(6, activation='relu'))
# model.add(Dense(3, activation='softmax'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(180, name='ys1')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(90, name='ys2')(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(46, name='ys3')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(22, name='ys4')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(6, name='ys5')(drop4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 13)]              0

#  ys1 (Dense)                 (None, 180)               2520

#  dropout (Dropout)           (None, 180)               0

#  ys2 (Dense)                 (None, 90)                16290

#  dropout_1 (Dropout)         (None, 90)                0

#  ys3 (Dense)                 (None, 46)                4186

#  dropout_2 (Dropout)         (None, 46)                0

#  ys4 (Dense)                 (None, 22)                1034

#  dropout_3 (Dropout)         (None, 22)                0

#  ys5 (Dense)                 (None, 6)                 138

#  dense (Dense)               (None, 3)                 21

# =================================================================
# Total params: 24,189
# Trainable params: 24,189
# Non-trainable params: 0



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



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_09',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
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
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
print("걸린 시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")


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