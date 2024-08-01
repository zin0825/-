
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
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


### 원핫 y 1-1 케라스 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)   # (50000, 100) (10000, 100)

#2. 모델
model = Sequential()
model.add(Conv2D(100, (3,3), activation='relu', input_shape=(32,32,3),
                 strides=2,
                 padding='same'))
model.add(Conv2D(filters=70, kernel_size=(2,2), activation='relu',
                 strides=2))
model.add(Conv2D(50, (2,2), activation='relu'))

model.add(Flatten())

model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Dense(100, activation='softmax'))


#3. 컴파일,  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=100, verbose=1,
                   restore_best_weights=True)


start = time.time()

hist = model.fit(x_train, y_train, epochs=15, batch_size=500,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

print('로스 : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1) 

accuracy = accuracy_score(y_test, y_predict) 
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)


# stride
# 로스 :  [3.066443920135498, 0.24619999527931213]
# acc_score :  0.2462

# 로스 :  [2.802659273147583, 0.310699999332428]
# acc_score :  0.3107