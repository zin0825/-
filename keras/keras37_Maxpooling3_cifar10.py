
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


###### 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0


###### OneHot
ohe =OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(32,32,3),
                 strides=1,
                 padding='same'))

model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation='relu',
                 strides=1,
                 padding='valid'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(units=16))
model.add(Dropout(0.3))
model.add(Dense(units=10, input_shape=(16,)))   # input_shape는 꼭 안해도 됨. 만약 할 경우 덴스랑 맞춰야함
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)


start = time.time()
hist = model.fit(x_train, y_train, epochs=15, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('로스 : ', loss)
print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print(y_pred)

accureacy = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('acc_score : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)



# acc_score :  0.394
# 걸린 시간 :  18.47 초
# 로스 :  [1.7561897039413452, 0.39399999380111694]

# acc_score :  0.6755
# 걸린 시간 :  67.0 초
# 로스 :  [0.9785659909248352, 0.6754999756813049]


# stride
# acc_score :  0.7289
# 걸린 시간 :  35.45 초
# 로스 :  [0.7947065830230713, 0.7289000153541565]


# MaxPooling2D / valid
# acc_score :  0.5455
# 걸린 시간 :  40.32 초
# 로스 :  [1.7251884937286377, 0.5454999804496765]

# acc_score :  0.6705
# 걸린 시간 :  43.68 초
# 로스 :  [0.9671040177345276, 0.6704999804496765]

# same
# acc_score :  0.7224
# 걸린 시간 :  43.09 초
# 로스 :  [0.8071021437644958, 0.7224000096321106]