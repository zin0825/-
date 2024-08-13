
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Reshape
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt




#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)   
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.2,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=15,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

augment_size =  40000   # 변현된 이미지도 같이 학습시키겠다 그래서 증폭함

print(x_train.shape[0])   # 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint 추출해서 증폭시키는 것
print(randidx)   # [ 6073 22404 11473 ... 41263 39534 49840]

print(x_train[0].shape)   # (32, 32, 3)

x_augmented = x_train[randidx].copy()   # 5만개 중에 4만개만 랜덤으로 가져옴 / randidx에 있는 인덱스의 x_train을 복사
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)   # (40000, 32, 32, 3) (40000, 1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]

print(x_augmented.shape)   # (40000, 32, 32, 3)

print(x_train.shape, x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented))    # shape 맞춰줘야 좋음
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)   # (90000, 32, 32, 3) (90000, 1)

### 원핫 y 1-1 케라스 
y_train = to_categorical(y_train)   # 다중분류여서 원핫인코딩을 해줌
y_test = to_categorical(y_test)   # 모양을 펴준다 / 0,1,2,3,4,5,6,7,8,9 -> 0과 1사이의 값으로 바꿔줌


x_train = x_train.reshape(90000,32,32*3)   # 4차원 -> 3차원으로 변경 / 위치 중요함 concatenate 얘 밑에 있어야함
x_test = x_test.reshape(10000,32,32*3)

print(x_train.shape, x_test.shape)   #(90000, 32, 96) (10000, 32, 96)

# exit()

# # # #2. 모델
model = Sequential()
# # model.add(Reshape(target_shape=(32 * 32))) 
model.add(LSTM(128, input_shape=(32, 32 * 3))) 
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(100, activation='softmax'))



# model.add(LSTM(64, input_shape=(32, 32))) 
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(36))
# model.add(Dense(10, activation='softmax'))


# # input1 = Input(shape=(32,32,3))   # = input_dim = (3,)))
# # maxpool1 = MaxPooling2D(2,2)(input1)
# # convd1 = Conv2D(64, (2,2), activation='relu', name='ys1')(maxpool1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
# # convd2 = Conv2D(128, (3,3), activation='relu', strides=1, 
# #                 padding='valid',name='ys2')(convd1)
# # maxpool2 = MaxPooling2D(2,2)(convd2)
# # drop1 = Dropout(0.2)(maxpool2)
# # convd3 = Conv2D(128, (2,2), activation='relu', strides=1,
# #                 padding='same', name='ys3')(drop1) 
# # drop2 = Dropout(0.2)(convd3)
# # maxpool2 = MaxPooling2D(2,2)(drop2)
# # drop3 = Dropout(0.2)(maxpool2)
# # convd4 = Conv2D(128, (2,2), activation='relu', name='ys4')(drop3)
# # drop4 = Dropout(0.2)(convd4)
# # flatten = Flatten()(drop4)
# # dense1 = Dense(36, name='ys5')(flatten)
# # output1 = Dense(100, activation='softmax')(dense1)
# # model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함

# model.summary()



#3. 컴파일,  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

start = time.time()

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
filepath = "".join([path, 'k59_17_01_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)

hist = model.fit(x_train, y_train, epochs=15, batch_size=188,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp]) 

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('acc_score : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)


# acc_score :  0.3084
# 걸린 시간 :  70.78 초
# 로스 :  [2.707254409790039, 0.3084000051021576]


# stride
# acc_score :  0.325
# 걸린 시간 :  33.53 초
# 로스 :  [2.638948917388916, 0.32499998807907104]

# acc_score :  0.361
# 걸린 시간 :  29.32 초
# 로스 :  [2.533759117126465, 0.3610000014305115]

# acc_score :  0.4549
# 걸린 시간 :  44.36 초
# 로스 :  [2.0676424503326416, 0.45489999651908875]


# acc_score :  0.3567
# 걸린 시간 :  29.05 초
# 로스 :  [2.5708413124084473, 0.35670000314712524]


# hamsu
# acc_score :  0.3567
# 걸린 시간 :  24.23 초
# 로스 :  [2.564917802810669, 0.35670000314712524]


# augment
# acc_score :  0.3199
# 걸린 시간 :  38.03 초
# 로스 :  [2.7397091388702393, 0.3199000060558319]


# LSTM
# acc_score :  0.0755
# 걸린 시간 :  33.25 초
# 로스 :  [4.05252742767334, 0.0754999965429306]
# k59_17_

# acc_score :  0.0718
# 걸린 시간 :  33.56 초
# 로스 :  [4.08433723449707, 0.07180000096559525]
# k59_17_01_