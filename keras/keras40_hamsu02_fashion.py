

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(x_train.shape, y_train.shape)   # (60000, 28, 28, 1) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000, 10)


##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   # 1.0 0.0


###### OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델
# model = Sequential()
# model.add(Conv2D(164, (3,3), input_shape=(28,28,1),
#                  strides=1,
#                  padding='same'))
# model.add(MaxPooling2D())
          
# model.add(Conv2D(filters=64, kernel_size=(3,3),
#                  strides=1))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3,3)))
# model.add(Dropout(0.5))
# model.add(Flatten())

# model.add(Dense(units=32))
# model.add(Dropout(0.4))
# model.add(Dense(units=10, input_shape=(32,)))
# model.add(Dense(units=10, activation='softmax'))




input1 = Input(shape=(28, 28, 1))   # = input_dim = (3,)))
convd1 = Conv2D(84, (3,3), strides=1, padding='same', name='ys1')(input1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
maxpool1 = MaxPooling2D((2,2))(convd1)
convd2 = Conv2D(filters=84, kernel_size=(3,3), 
                strides=1, padding='valid', name='ys2')(maxpool1)
drop1 = Dropout(0.5)(convd2)
convd3 = Conv2D(64, (3,3), name='ys3')(drop1) 
drop1 = Dropout(0.5)(convd3)
flatten = Flatten()(drop1)
dense1 = Dense(16, name='ys4')(flatten)
drop2 = Dropout(0.4)(dense1)
dense2 = Dense(10, name='ys5')(drop2)
output1 = Dense(10)(dense2)
model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함
model.summary()






#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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



# acc_score :  0.831
# 걸린 시간 :  17.3 초
# 로스 :  [0.49016499519348145, 0.8309999704360962]


# stride
# acc_score :  0.829
# 걸린 시간 :  12.97 초
# 로스 :  [0.478302538394928, 0.8289999961853027]

# acc_score :  0.8341
# 걸린 시간 :  34.33 초
# 로스 :  [0.4714318811893463, 0.8341000080108643]

# MaxPooling2D / valid
# acc_score :  0.8868
# 걸린 시간 :  27.74 초
# 로스 :  [0.31969863176345825, 0.8867999911308289]

# acc_score :  0.9614
# 걸린 시간 :  5.61 초

# same
# acc_score :  0.8912
# 걸린 시간 :  30.53 초
# 로스 :  [0.30836427211761475, 0.8912000060081482]


# hamsu
# acc_score :  0.1824
# 걸린 시간 :  21.53 초
# 로스 :  [3.068488121032715, 0.18240000307559967]