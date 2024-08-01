
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
model.add(Conv2D(64, (2,2), activation='relu', input_shape=(32,32,3),
                 strides=1,
                #  padding='same'
                ))     
model.add(MaxPooling2D(pool_size=3, padding='same'))   
# 모르는거 있으면 ()안에 컨트롤 + 스페이스 하면 됨/ 가로세로를 다시 잡고 싶은면 이걸 하면 됨

model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                 strides=1,
                 padding='valid'))
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu',
                 strides=1,
                 padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu',
                 strides=1,
                 padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(36))
model.add(Dense(100, activation='softmax'))

model.summary()



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


# acc_score :  0.3084
# 걸린 시간 :  70.78 초
# 로스 :  [2.707254409790039, 0.3084000051021576]


# stride
# acc_score :  0.325
# 걸린 시간 :  33.53 초
# 로스 :  [2.638948917388916, 0.32499998807907104]

# acc_score :  0.361
# 걸린 시간 :  29.32 초
# 로스 :  [2.533759117126465, 0.3610000014305115]

# acc_score :  0.4549
# 걸린 시간 :  44.36 초
# 로스 :  [2.0676424503326416, 0.45489999651908875]