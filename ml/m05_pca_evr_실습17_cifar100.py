
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
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


# #### 스케일링
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train))   # 1.0 0.0

# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)

# print(y_train.shape, y_test.shape)

# ### 원핫 y 1-1 케라스 //
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# #2. 모델
# model = Sequential()
# model.add(Dense(64, input_shape=(32*32*3,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(36, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# model.summary()



# #3. 컴파일,  훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)

# start = time.time()

# hist = model.fit(x_train, y_train, epochs=15, batch_size=188,
#                  verbose=1,
#                  validation_split=0.2,
#                  callbacks=[es]) 

# end = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)

# print('acc : ', round(loss[1],2))

# y_pred = model.predict(x_test)

# y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)

# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
# acc = accuracy_score(y_test, y_pred)
# print('acc_score : ', acc)
# print('걸린 시간 : ', round(end - start,2), "초")
# print('로스 : ', loss)



# acc_score :  0.4549
# 걸린 시간 :  44.36 초
# 로스 :  [2.0676424503326416, 0.45489999651908875]


# acc_score :  0.2046
# 걸린 시간 :  14.17 초
# 로스 :  [3.3856019973754883, 0.2046000063419342]