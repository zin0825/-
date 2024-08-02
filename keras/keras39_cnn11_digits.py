from sklearn.datasets import load_digits   # digits 숫자
import pandas as pd

x, y = load_digits(return_X_y=True)   # x와 y로 바로 반환해줌
print(x)
print(y)
print(x.shape, y.shape)   # (1797, 64) (1797,)   이미지는 0에서 225의 숫자를 부여함 225가 가장 진함 놈
# 1797장의 이미지가 있는데 8바이8 짜리를 64장으로 쭉 한것, 원래는 (1797,8,8)의 이미지 인데 칼라는 (1797,8,8,1)

print(pd.value_counts(y, sort=False))   # 확인, ascending=True 오름차순  # y라벨 10개
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import time
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
# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (1797, 10)


y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (1797, 10)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (1797, 10)


print(x.shape, y.shape)   # (1797, 64) (1797, 10)

x = x.reshape(1797, 64, 1, 1)

x = x/255.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)




# #2. 모델구성
# model = Sequential()
# model.add(Dense(180, activation='relu', input_dim=64))
# model.add(Dropout(0.6))
# model.add(Dense(90))
# model.add(Dropout(0.5))
# model.add(Dense(46))
# model.add(Dropout(0.4))
# model.add(Dense(6))
# model.add(Dense(10, activation='softmax'))


# #2-2. 모델구성(함수형)  
# input1 = Input(shape=(64,))  
# dense1 = Dense(180, name='ys1')(input1)  
# drop1 = Dropout(0.6)(dense1)
# dense2 = Dense(90, name='ys2')(drop1) 
# drop2 = Dropout(0.5)(dense2)
# dense3 = Dense(46, name='ys3')(drop2)
# drop3 = Dropout(0.4)(dense3)
# dense4 = Dense(6, name='ys4')(drop3)
# output1 = Dense(10, activation='softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)  
# model.summary()


model = Sequential()
model.add(Conv2D(180, (3,3), input_shape=(64,1,1), strides=1, activation='relu',padding='same')) 
model.add(Dropout(0.6))
model.add(Conv2D(90, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(46, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.4))
model.add(Flatten())                            

model.add(Dense(6, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))



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



path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k39_11',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)

hist = model.fit(x_train, y_train, epochs=180, batch_size=3086, 
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

accuracy_score = accuracy_score(y_test, y_pred)
print(y_pred)

print('acc score : ', accuracy_score)

r2 = r2_score(y_test, y_pred)
print('r2 score : ', r2)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)

    


# acc score :  0.9388888888888889
# 걸린 시간 :  6.83 초
# 로스 :  [0.17338207364082336, 0.9444444179534912]


# Dropout (0.6, 0.5, 0.4)
# acc score :  0.9388888888888889
# r2 score :  0.9074074074074074
# 걸린 시간 :  6.67 초
# 로스 :  [0.14785702526569366, 0.9611111283302307]   로스가 떨어져도 좋음

# (0.6, 0.6, 0.5)   너무 많이 하면 훈련이 안됨. 적당히
# acc score :  0.9444444444444444
# r2 score :  0.9074074074074074
# 걸린 시간 :  6.74 초
# 로스 :  [0.12595827877521515, 0.9555555582046509]


# acc score :  0.9055555555555556
# r2 score :  0.8333333333333334
# 걸린 시간 :  6.72 초
# 로스 :  [0.3185581564903259, 0.9166666865348816]
# 쥐피유 없다! xxxxx


# acc score :  0.9222222222222223
# r2 score :  0.8395061728395061
# 걸린 시간 :  7.86 초
# 로스 :  [0.24413441121578217, 0.9277777671813965]
# 쥐피유 돈다!!!


# dnn -> cnn
