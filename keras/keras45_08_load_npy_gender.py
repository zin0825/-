
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



start1 = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save_npy\\'

x_train = np.load(np_path + 'keras45_07_x_train.npy')   # load용
y_train = np.load(np_path + 'keras45_07_y_train.npy')



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, 
                                                    test_size=0.1, random_state=313)

print(x_train)
print(x_train.shape)   
print(y_train)
print(y_train.shape)  


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

path = './_save/keras45/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k45_08_', date, '_', filename])  


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)


start = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=8,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])

end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)



print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)



# #5. 파일 출력
# y_submit = model.predict(xy_test, batch_size=12)
# mission_csv['label'] = y_submit

# mission_csv.to_csv(path_mission + 'mission_0805_1007.csv')




