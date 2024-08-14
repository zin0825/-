"""
keras49_augment2_mnist.py
keras49_augment3_cifar10.py
keras49_augment4_cifar100.py
keras49_augment5_cat_dog.py
keras49_augment6_man_woman.py
keras49_augment7_horse.py
keras49_augment8_rpg.py
"""

# cat dog 은 image 폴더꺼 수치화(2만개)하고,
# 캐글 폴더꺼 수치화(2.5만개) 해서 합치고
# 증폭 5천개 추가   // (총 5.5만개)
#해서 맹그러 서 kaggel에 제출까지





import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,) 샘플만 요약해 나와서 000만 나옴
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)


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

augment_size =  40000

print(x_train.shape[0])   # 60000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)   # [16546 57499 17658 ... 45836 48290 52924]


print(x_train[0].shape)   # (28, 28)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)   # (40000, 28, 28) (40000,)


x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)
print(x_augmented.shape)   # (40000, 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]

print(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)   # (100000, 28, 28, 1) (100000,)




### 원핫 y 1-1 케라스 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델
model = Sequential()
model.add(Conv1D(filters=164, kernel_size=2, input_shape=(28, 28)))
model.add(Conv1D(164, 2))
model.add(Flatten()) 
model.add(Dropout(0.3))
model.add(Dense(units=132, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=62, activation='relu'))    
model.add(Dense(units=36, activation='relu'))    
model.add(Dropout(0.4))
model.add(Dense(units=10, activation='relu'))    
model.add(Dense(units=10, activation='sigmoid'))



# input1 = Input(shape=(28, 28, 1))   # = input_dim = (3,)))
# convd1 = Conv2D(164, (3,3), strides=1, padding='same', name='ys1')(input1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
# convd2 = Conv2D(164, (3,3), strides=1, 
#                 padding='same',name='ys2')(convd1)
# maxpool1 = MaxPooling2D((2,2))(convd2)
# drop1 = Dropout(0.3)(maxpool1)
# convd3 = Conv2D(132, (2,2), name='ys3')(drop1) 
# drop2 = Dropout(0.5)(convd3)
# flatten = Flatten()(drop2)
# dense1 = Dense(62, name='ys4')(flatten)
# dense2 = Dense(36, name='ys5')(dense1)
# drop3 = Dropout(0.4)(dense2)
# dense3 = Dense(10)(drop3)
# output1 = Dense(10)(dense3)
# model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함

# model.summary()


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])   # metrics 정확도 확인

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
                   restore_best_weights=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=1228,   
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.hdf5')   # 두가지 다 저장할거야

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)
print('acc : ', round(loss[1],2))

y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)   # 회귀가 아니기 때문에 acc가 더 정확함
# print('r2 score : ', r2)


y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)   # 원핫인코딩한 애들을 다시 원핫인코딩 하기 전으로 변환

print(y_predict)

accuracy = accuracy_score(y_test, y_predict)   # 변수에 (초기화 안된) 변수를 넣어서 오류 뜸 
acc = accuracy_score(y_test, y_predict) # 예측한 y값과 비교
print("acc_score : ", acc)
print("걸린 시간 : ", round(end - start,2),'초')
print('로스 : ', loss)


# 로스 :  [0.3007262349128723, 0.9153000116348267]
# acc :  0.92
# acc_score :  0.9153
# 걸린 시간 :  21.72 초


# stride
# acc_score :  0.9628
# 걸린 시간 :  5.88 초


# maxpooling / valid
# acc_score :  0.9597
# 걸린 시간 :  5.61 초

# same
# acc_score :  0.9676
# 걸린 시간 :  11.68 초


# hamsu
# acc_score :  0.3154
# 걸린 시간 :  73.41 초
# 로스 :  [2.099454402923584, 0.31540000438690186]


# augmente
# acc_score :  0.202
# 걸린 시간 :  159.48 초
# 로스 :  [2.1953086853027344, 0.20200000703334808]