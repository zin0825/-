
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 파이썬 버전 달라서 없으면 찾을 것. 분명 어딘가에 있음
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()   
# 판다스로 데이터 프레임을 자유자재로 쓰는게 중요. 데이터셋 변경하는거 확실히 배워둘 것
# x_train 사이즈 = (60000, 28, 28)

x_train = x_train/255.
x_test = x_test/255.   # 이거 할거면 밑에거 지워야함

train_datagen = ImageDataGenerator(
    # rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.2,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=15,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

augment_size = 40000   # 아그먼트. 증가시키다. 변수명도 이름 잘 생각하면서 만들것

print(x_train.shape[0])   # 60000 / 0 =60000, 1 =28, 2 =28, 3 =x 본 이미지 사이즈 그때마다 다름
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 0번째 28,28,1/ 60000, size=4000
print(randidx)   # [57909 18590 26842 ... 12567   523 44430]
# 중복 데이터라서 변환 시켜야함

# 지금까지는 예시이고 실질적으로 사용하는 건 49_1번. 외우기!!! 계속 쓸거임 (증폭)

print(np.min(randidx), np.max(randidx))   # 2 59999, 1 59998 랜덤이라 다름 60000만 안넘으면 됨

print(x_train[0].shape)   # (28, 28)

x_augmented = x_train[randidx].copy()   # 메모리를 별도로 잡는다. 원본 x_train에 영향을 주지 않는다. 메모리 안전빵
y_augmented = y_train[randidx].copy()   # x와 y의 순서가 바뀌면 안됨!
print(x_augmented.shape, y_augmented.shape)   # (40000, 28, 28) (40000,)

# x_augmented = x_augmented.reshape(40000, 28, 28, 1)   # 쉐이프를 명확하게 안다면 이렇게 해도 상관 xx 하지만 밑에걸로 진행
x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0],   
                                  # 40000 새로 (증폭시켜) 만든 augmented 사이즈
                                  x_augmented.shape[1],   # 28
                                  x_augmented.shape[2], 1)   # 28, 1
# print(x_augmented.shape)   # (40000, 28, 28, 1)


x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_tset = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28)

x_train = np.concatenate((x_train, x_augmented))   # () 하나만 하면 에러남 (()) 두개 해야함
y_train = np.concatenate((y_train, y_augmented))   # axis=0은 디폴트 생략가능

print(x_train.shape, y_train.shape)   # (100000, 28, 28, 1) (100000,)


#### 맹그러봐 ####


### 원핫 y 1-1 케라스 
y_train = to_categorical(y_train)   # 다중분류여서 원핫인코딩을 해줌
y_test = to_categorical(y_test)



#2. 모델
model = Sequential()
model.add(LSTM(84, input_shape=(28, 28))) 
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))


# input1 = Input(shape=(28, 28, 1))   # = input_dim = (3,)))
# convd1 = Conv2D(84, (3,3), strides=1, padding='same', name='ys1')(input1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
# maxpool1 = MaxPooling2D((2,2))(convd1)
# convd2 = Conv2D(filters=84, kernel_size=(3,3), 
#                 strides=1, padding='valid', name='ys2')(maxpool1)
# drop1 = Dropout(0.5)(convd2)
# convd3 = Conv2D(64, (3,3), name='ys3')(drop1) 
# drop2 = Dropout(0.5)(convd3)
# flatten = Flatten()(drop2)
# dense1 = Dense(16, name='ys4')(flatten)
# drop3 = Dropout(0.4)(dense1)
# dense2 = Dense(10, name='ys5')(drop3)
# output1 = Dense(10)(dense2)
# model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
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
filepath = "".join([path, 'k59_15_01_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)

hist = model.fit(x_train, y_train, epochs=15, batch_size=298,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es, mcp])
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)
print('acc : ', round(loss[1],2))

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)   # 원핫인코딩한 애들을 다시 원핫인코딩 하기 전으로 변환

print(y_predict)

accuracy = accuracy_score(y_test, y_predict)   # 변수에 (초기화 안된) 변수를 넣어서 오류 뜸 
acc = accuracy_score(y_test, y_predict) # 예측한 y값과 비교
print("acc_score : ", acc)
print("걸린 시간 : ", round(end - start,2),'초')
print('로스 : ', loss)


# hamsu
# acc_score :  0.1824
# 걸린 시간 :  21.53 초
# 로스 :  [3.068488121032715, 0.18240000307559967]

# augment
# acc_score :  0.3323
# 걸린 시간 :  35.39 초
# 로스 :  [2.719804048538208, 0.33230000734329224]


# LSTM
# acc_score :  0.868
# 걸린 시간 :  22.89 초
# 로스 :  [0.3672025501728058, 0.8679999709129333]
# k59_15_

# _score :  0.8653
# 걸린 시간 :  22.77 초
# 로스 :  [0.37850892543792725, 0.8652999997138977]
# k59_15_01_
