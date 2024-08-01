

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)   # 샘플만 요약해 나와서 000만 나옴
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  6만의 28, 28, 1 / 컬러는 6만의 28, 28, 3
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)   만의 28, 28, 1


x_train = x_train.reshape(60000, 28, 28, 1)   # 1이 없으면 2차원이니까 3차원으로 정의하기 위해
x_test = x_test.reshape(10000, 28, 28, 1)


# y_train = pd.get_dummies(y_train)   # 이걸 해야 10이 생김
# y_test = pd.get_dummies(y_test)

# print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)


# ##### 스켈일링 1-1
x_train = x_train/255.   # 소수점
x_test = x_test/255.

# print(np.max(x_train), np.min(x_train))   # 1.0 0.0


###### 스케일링 1-2
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_train), np.min(x_train))   # 1.0 -1.0   스케일링 두번하면 문제 있으니 주의
# 두 개 돌려봐서 좋은거 쓸 것

###### 스케일링 2. MinMaxScaler(), StandardScaler()
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))   # 1.0000000000000002 0.0


# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, x_test.shape)


# ### 원핫 y 1-1 케라스 // 판다스, 사이킷런으로도 맹그러  항상 시작은 0부터다
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# # ### 원핫 y 1-2 판다스
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# # print(y_train.shape, y_test.shape)
# # # 여기까지가 #1 완성 --------------------


### 원핫 y 1-3 사이킷런
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1),
                 strides=1,
                 padding='same'))  # 26, 26, 64
                            # shape = (batch_size, rows, columns, channels)   # 왜 batch_size인가 훈련의 데이터를 배치사이즈 단위로 한다.
                                                                              # 행의 갯수는 중요하지 않음 전체에서 얼마만큼 잘라 쓰느냐
                            # shape = (batch_size, heights, widths, channels)   


model.add(Conv2D(filters=64, kernel_size=(3,3),
                 strides=1,
                 padding='same'))   # 24, 24, 64
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(32, (2,2), activation='relu', strides=1))   # 23, 23, 32
model.add(Dropout(0.5))
model.add(Flatten())   # 23 * 23 * 32

model.add(Dense(units=32))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10, input_shape=(32,)))
                         # shape = (batch_size, input_dim)

model.add(Dense(10, activation='softmax'))   # 0부터 9까지 y를 맞추기 위해

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
hist = model.fit(x_train, y_train, epochs=5, batch_size=1228,   
          verbose=1, 
          validation_split=0.2,
          callbacks=[es]
)
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