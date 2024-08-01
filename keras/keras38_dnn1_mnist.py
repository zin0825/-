
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)   # 샘플만 요약해 나와서 000만 나옴
print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  6만의 28, 28, 1 / 컬러는 6만의 28, 28, 3
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)   만의 28, 28, 1

# x는 6만의 28, 28, 1로 리쉐이프
# y는 원핫앤코딩


x_train = x_train.reshape(60000, 28, 28, 1)   # 1이 없으면 2차원이니까 3차원으로 정의하기 위해
x_test = x_test.reshape(10000, 28, 28, 1)



# y_train = pd.get_dummies(y_train)   # 이걸 해야 10이 생김
# y_test = pd.get_dummies(y_test)
# print(y_train)   # [60000 rows x 10 columns]
# print(y_test)   # [10000 rows x 10 columns]

# print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)


# ##### 스켈일링 1-1   # 이 데이터는 이 안에 넣어줘야겠다
x_train = x_train/255.   # 소수점
x_test = x_test/255.

# print(np.max(x_train), np.min(x_train))   # 1.0 0.0


x_train = x_train.reshape(60000, 28 * 28)   # 흑백 1은 묵음
x_test = x_test.reshape(10000, 28 * 28)

print(y_train.shape, y_test.shape)


### 원핫 y 1-1 케라스 //
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# # # 여기까지가 #1 완성 --------------------


#2. 모델
model = Sequential()
model.add(Dense(128, input_shape=(28*28,)))   # 흑백 1은 묵음
# 맹그러봐
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(44, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

start = time.time()

hist = model.fit(x_train, y_train, epochs=15, batch_size=188,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es]) 

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


# acc_score :  0.9758
# 걸린 시간 :  18.6 초
# 로스 :  [0.10267635434865952, 0.9757999777793884]






