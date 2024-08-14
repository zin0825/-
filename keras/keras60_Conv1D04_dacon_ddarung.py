

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터
path = './_data/dacon/따릉이/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)   

test_csv = pd.read_csv(path + "test.csv", index_col=0)  

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)  
    
print(train_csv.shape)   # (1459, 10)
print(test_csv.shape)   # (715, 9) 
print(submission_csv.shape)   # (715, 1)

print(train_csv.columns) 

print(train_csv.info())
print(train_csv.isna().sum()) 

train_csv = train_csv.dropna()
print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) 

# print(test_csv.info())  # (715, 9)

x = train_csv.drop(['count'], axis=1)    
print(x)    # [1328 rows x 9 columns]

y = train_csv['count'] 
# print(y.shape)    # (1328,)
x = x.to_numpy()

x = x.reshape(1328, 9, 1)
# x = x/255.

print(x.shape, y.shape)   # (1328, 3, 3, 1) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)

print(x.shape, y.shape)   # (1328, 9) (1328,)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=500, kernel_size=2, input_shape=(9,1)))   # input_shape=(3,1) 행무시
model.add(Conv1D(250, 2))
# model.add(MaxPooling2D())
model.add(Conv1D(125, 2))        
model.add(Flatten())                            

model.add(Dense(units=79, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)


######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))  
date = date.strftime("%m%d_%H%M")
# print(date)     
# print(type(date))  

path = './_save/keras60/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k60_04_01_', date, '_', filename])  


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)



hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1, validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0) 

print('acc : ', round(loss[1],2))

print("걸린 시간 :", round(end-start,2),'초')
print("로스 : ", loss)


# 로스 :  2259.459228515625
# r2스코어 :  0.6430264394639782
# 걸린시간 :  2.03 초
# 쥐피유 없다! xxxxx

# 로스 :  2264.621826171875
# r2스코어 :  0.6422108244203191
# 걸린시간 :  3.31 초
# 쥐피유 돈다!!!



# dnn -> cnn
# 걸린 시간 : 14.84 초
# 로스 :  [2063.5146484375, 0.0]


# Conv1D
# acc :  0.0
# 걸린 시간 : 19.47 초
# 로스 :  [2278.81689453125, 0.0]
# k60_04_

# acc :  0.0
# 걸린 시간 : 9.95 초
# 로스 :  [2297.495361328125, 0.0]
# k60_04_01_