
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_train.shape)   # (50000, 32, 32, 3) (50000, 32, 32, 3)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


#### 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0


#### OneHot 
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(78, (6,6), activation='relu', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(100, activation='softmax'))


#3. 컴파일,  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

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



path = './_save/keras35/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k35_07',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1,
                      save_best_only=True, filepath=filepath)

start = time.time()

hist = model.fit(x_train, y_train, epochs=15, batch_size=128,
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
