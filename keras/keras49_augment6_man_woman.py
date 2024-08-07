
# 모델 구성해서 가중치까지 세이브할것


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import natsort





np_path = 'C:\\프로그램\\ai5\\_data\\_save_npy\\'

x_train = np.load(np_path + 'keras45_07_x_train.npy')   # load용
y_train = np.load(np_path + 'keras45_07_y_train.npy')


train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=5,   # 정해진 각도만큼 이미지 회전
    zoom_range=1.5,   # 축소 또는 화대, 1.2배
    shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

start1 = time.time()




x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                    test_size=0.1, random_state=313)


print(x_train.shape)   # (24450, 100, 100, 3)
print(y_train.shape)   # (24450,)

print(x_test.shape)   # (2717, 100, 100, 3)
print(y_test.shape)   # (2717,)


augment_size =  10000

print(x_train.shape[0]) 

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)   # [12411 10225  4238 ...  9569  6497 19205]
print(np.min(randidx), np.max(randidx))    # 1 24449
print(x_train[0].shape)   # (100, 100, 3)


x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)   # (10000, 100, 100, 3) (10000,)


x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0],   # 
                                  x_augmented.shape[1],   # 
                                  x_augmented.shape[2], 3)   # 컬러라서 , 3
print(x_augmented.shape)   # (10000, 100, 100, 3)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                 save_to_dir='').next()[0]

print(x_augmented.shape) 






"""
x_train = x_train.reshape(24450, 100, 100, 3)
x_test = x_test.reshape(2717, 100, 100, 3)

print(x_train.shape, x_test.shape)   # (24450, 100, 100, 3) (2717, 100, 100, 3)


x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)   # (34450, 100, 100, 3) (34450,)


end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1,2), "초")

#3. 모델 구성
model = Sequential()
model.add(Conv2D(24, (2,2), input_shape=(100, 100, 3), activation='relu',
                 strides=1, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=24, kernel_size=(2,2), activation='relu', 
                 strides=1, padding='same'))

model.add(Conv2D(20, (2,2), strides=1, padding='same'))
model.add(Conv2D(20, (2,2), strides=1, padding='same'))
model.add(Conv2D(20, (2,2), strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)


######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))  
date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

path = './_save/keras49/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k49_06_', date, '_', filename])  


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)


start = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=6,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])

end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=8)

y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)



print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)

"""

# #5. 파일 출력
# y_submit = model.predict(y_test, batch_size=12)
# mission_csv['label'] = y_submit

# mission_csv.to_csv(path_mission + 'mission_0805_1007.csv')


# 걸린 시간1 :  12.48 초
# acc :  0.6569746043430253
# 걸린 시간 :  209.0 초
# 로스 :  [0.5989478230476379, 0.6569746136665344]


# 걸린 시간1 :  12.18 초
# acc :  0.6569746043430253
# 걸린 시간 :  282.94 초
# 로스 :  [0.5837536454200745, 0.6569746136665344]