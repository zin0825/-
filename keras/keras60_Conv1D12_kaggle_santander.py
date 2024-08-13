# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits  


import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]



#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)   # (200000, 201)
# print(test_csv.shape)   # (200000, 200)
# print(sample_csv.shape)   # (200000, 1)


print(train_csv.isna().sum())
# print(train_csv.columns)

train_csv = train_csv.dropna()
print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) 

x = train_csv.drop(['target'], axis=1)
# print(x)   # [200000 rows x 200 columns]

y = train_csv['target']

print(x.shape, y.shape)   # (200000, 200) (200000,)


x = x.to_numpy()

x = x.reshape(200000, 200, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038,
                                                    stratify=y)



# #2. 모델구성
model = Sequential()
model.add(LSTM(500, input_shape=(200, 1))) 
model.add(Dropout(0.6))
model.add(Dense(400))
model.add(Dense(340))
model.add(Dropout(0.6))
model.add(Dense(330))
model.add(Dense(250))
model.add(Dropout(0.6))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))


# #2-2. 모델구성(함수형)  
# input1 = Input(shape=(200,))  
# dense1 = Dense(500, name='ys1')(input1)  
# drop1 = Dropout(0.6)(dense1)
# dense2 = Dense(400, name='ys2')(drop1) 
# dense3 = Dense(340, name='ys3')(dense1) 
# drop2 = Dropout(0.6)(dense3)
# dense4 = Dense(330, name='ys4')(drop2)
# dense5 = Dense(250, name='ys5')(dense4)
# drop3 = Dropout(0.6)(dense5)
# dense6 = Dense(100, name='ys6')(drop3)
# dense7 = Dense(50, name='ys7')(dense3)
# output1 = Dense(1, activation='sigmoid')(dense7)
# model = Model(inputs=input1, outputs=output1)  
# model.summary()


# model = Sequential()
# model.add(Conv2D(200, (3,3), input_shape=(200, 1, 1), strides=1, activation='relu',padding='same')) 
# model.add(Conv2D(500, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(Dropout(0.6))
# model.add(Conv2D(400, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Conv2D(340, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Dropout(0.6))
# model.add(Flatten())                            

# model.add(Dense(330, activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.add(Dense(units=3, activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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



path = './_save/keras59/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k59_12_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################
 

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath=filepath)
    
    
start = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=1000, 
                 verbose=1, 
                 validation_split=0.3,
                 callbacks=[es, mcp])
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)



y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
sample_csv['target'] = y_submit
"""sample_csv.to_csv(path + "sample_submission_0724_1642.csv")"""
accuracy_score = accuracy_score(y_test, y_pred)

print('acc score : ', accuracy_score)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)





# acc score :  0.87495
# 걸린 시간 :  104.54 초
# 로스 :  [1.0752360820770264, 0.8749499917030334]

# batch_size=1000
# acc score :  0.897375
# 걸린 시간 :  117.65 초
# 로스 :  [0.3148635923862457, 0.8973749876022339]

# batch_size=5000
# acc score :  0.9049
# 걸린 시간 :  58.69 초
# 로스 :  [0.25515782833099365, 0.9049000144004822]

# acc score :  0.91195
# 걸린 시간 :  60.1 초
# 로스 :  [0.24282044172286987, 0.9119499921798706]

# acc score :  0.91185
# 걸린 시간 :  58.62 초
# 로스 :  [0.24183888733386993, 0.9118499755859375]

# 스켈링
# acc score :  0.912525
# 걸린 시간 :  120.47 초
# 로스 :  [0.2372458279132843, 0.9125249981880188]


# acc score :  0.913775
# 걸린 시간 :  122.54 초
# 로스 :  [0.23384465277194977, 0.9137750267982483]


# save
# acc score :  0.914975
# 걸린 시간 :  34.86 초
# 로스 :  [0.2327820509672165, 0.9149749875068665]


# dropout
# acc score :  0.911025
# 걸린 시간 :  44.47 초
# 로스 :  [0.2387518286705017, 0.9110249876976013]


# acc score :  0.914275
# 걸린 시간 :  20.08 초
# 로스 :  [0.23369953036308289, 0.9142749905586243]
# 쥐피유 없다! xxxxx

# acc score :  0.913975
# 걸린 시간 :  11.1 초
# 로스 :  [0.2337251454591751, 0.9139750003814697]
# 쥐피유 돈다!!!
