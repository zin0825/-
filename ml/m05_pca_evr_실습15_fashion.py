

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])
print("y_train[0] : ", y_train[0])


print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)

x_train = x_train.reshape(60000, 28, 28, 1)   # 1이 없으면 2차원이니까 3차원으로 정의하기 위해
x_test = x_test.reshape(10000, 28, 28, 1)


##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0

x_train = x_train.reshape(60000, 28 * 28) 
x_test = x_test.reshape(10000, 28 * 28)

print(y_train.shape, y_test.shape)


### 원핫 y 1-1 케라스 //
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델
model = Sequential()
model.add(Dense(164, input_shape=(28*28,)))
model.add(Dense(164, activation='relu'))
model.add(Dense(104, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10,   # patience=참을성
                   verbose=1,   
                   restore_best_weights=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=15, batch_size=298,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])
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


# same
# acc_score :  0.8912
# 걸린 시간 :  30.53 초
# 로스 :  [0.30836427211761475, 0.8912000060081482]

# acc_score :  0.8655
# 걸린 시간 :  10.36 초
# 로스 :  [0.37805598974227905, 0.8654999732971191]