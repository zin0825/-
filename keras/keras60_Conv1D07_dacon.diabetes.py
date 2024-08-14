
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
path = "./_data/dacon/biabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]
print(x)    # [652 rows x 8 columns]
print(y.shape)    # (652, )

x = x.to_numpy()
x = x.reshape(652, 4, 2)

x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=7575)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, input_shape=(4, 2)))
model.add(Conv1D(64, 2))
model.add(Dense(units=148, activation='relu'))
model.add(Flatten())                            

model.add(Dense(units=168, activation='relu'))
model.add(Dropout(0.2))    
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))



# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(2,2,2), strides=1, padding='same')) 
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(148, (3,3), activation='relu', strides=1, padding='same'))        
# model.add(Flatten())                            

# model.add(Dense(units=168, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True )

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime
date = datetime.datetime.now()
print(date)    
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date)     
print(type(date))  

path = './_save/keras60/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k60_07_', date, '_', filename]) 

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print("ACC : ", round(loss[1], 3))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print(y_predict[:20])
y_predict = np.round(y_predict)
print(y_predict[:20])

accuracy_score = accuracy_score(y_test, y_predict)
print(y_predict)

print("로스 : ", loss[0])
print("r2 : ", r2)
print("걸린시간 : ", round(end - start, 2), "초")

# dnn -> cnn
# 로스 :  0.5475181341171265
# r2 :  0.09402174678600772
# 걸린시간 :  8.42 초


# Conv1D
# 로스 :  0.5340856909751892
# r2 :  0.13282125561363345
# 걸린시간 :  5.39 초
# k60_07_
