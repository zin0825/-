# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


#1. 데이터
path = ".\\_data\\keggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(mission_csv.shape)    # (110023, 1)

print(train_csv.isna().sum())



le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x = x.to_numpy()
x = x.reshape(165034, 5, 2, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, 
                                                    random_state=7680)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', input_shape=(5,2,1), strides=1, padding='same')) 
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(148, (2,2), activation='relu', strides=1, padding='same'))    
model.add(Dropout(0.3))
model.add(Conv2D(168, (2,2), activation='relu', strides=1, padding='same'))        
model.add(Dropout(0.3))
model.add(Flatten())                            

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime
date = datetime.datetime.now()
print(date)    
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date)     
print(type(date))  

path = './_save/keras39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k39_08_', date, '_', filename])     

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)

hist = model.fit(x_train, y_train, epochs=2680, batch_size=6220, 
          verbose=1, 
          validation_split=0.2,
          callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("걸린 시간 : ", round(end - start, 2), "초")

print("로스 : ", loss)

"""
y_submit = model.predict(test_csv)
print(y_submit.shape)       # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_0729_1407.csv")
"""


# acc score :  0.7877419941224588
# 걸린 시간 :  18.89 초
# 로스 :  [0.4583592712879181, 0.787742018699646]
# 쥐피유 없다! xxxxx


# acc score :  0.7877116975187082
# 걸린 시간 :  4.98 초
# 로스 :  [0.45404523611068726, 0.7877116799354553]
# 쥐피유 돈다!!!


# dnn -> cnn
# acc score :  0.7877116975187082
# 걸린 시간 :  11.85 초
# 로스 :  [0.5103834867477417, 0.7877116799354553]

