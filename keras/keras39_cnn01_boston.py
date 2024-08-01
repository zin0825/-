# CNN으로 맹그러

"""
01. 보스톤
02. california
03. diabetes
04. dacon_ddarung
05. kaggle_bike

06_cancer
07_dacon_diabetes
08_kaggle_bank
09_wine
10_fetch_covtpe
11_digits
"""


import sklearn as sk

from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




#1. 데이터 #1. 데이터 
dataset = load_boston()

x = dataset.data   
y = dataset.target  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)



print(x_train.shape, y_train.shape)   # (404, 13) (404,)   # 무슨 값이 알고싶어
print(x_test.shape, y_test.shape)   # (102, 13) (102,)


x_train = x_train.reshape(404, 13, 1, 1)   # 정의된 값을 변환할거야
x_test = x_test.reshape(102, 13, 1, 1)

print(x_train, x_test)

x_train = x_train/255.
x_test = x_test/255.




#2. 모델 구성
model = Sequential()
model.add(Conv2D(180, (2,1), input_shape=(13,1,1)))
model.add(Conv2D(128, (2,1), activation='relu'))
model.add(Conv2D(100, (2,1), activation='relu'))
model.add(Flatten())

model.add(Dense(96, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True)


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=16,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es])

end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1)

print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2_score : ', r2)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)


# r2_score :  0.7912355514442061
# 걸린 시간 :  15.51 초
# 로스 :  [15.604340553283691, 0.0]