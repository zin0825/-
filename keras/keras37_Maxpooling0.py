

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
model.add(Conv2D(10, (3,3), input_shape=(28, 28, 1),
                # strides=1,
                #  padding='same')  # 26, 26, 10 
                padding='valid'))   # valid 디폴트
model.add(MaxPooling2D())   # 13, 13, 10   이미지만 반토막

model.add(Conv2D(filters=9, kernel_size=(3,3),   # 11, 11, 9
                 strides=1,
                 padding='valid'))   # 24, 24, 64
model.add(Dropout(0.5))
model.add(Conv2D(8, (2,2), activation='relu'))   # 10, 10,8
model.add(Dropout(0.5))

model.add(Dense(units=32))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10, input_shape=(32,)))
                         # shape = (batch_size, input_dim)

model.add(Dense(10, activation='softmax'))   # 0부터 9까지 y를 맞추기 위해

model.summary()

# # # model.summary()
# # # # _________________________________________________________________
# # # #  Layer (type)                Output Shape              Param #
# # # # =================================================================
# # # #  conv2d (Conv2D)             (None, 27, 27, 10)        50

# # # #  conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820

# # # #  conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815

# # # #  flatten (Flatten)           (None, 7260)              0   # 왜 0인가? 모양만 (펴준기만) 바뀐거라 0이다

# # # #  dense (Dense)               (None, 8)                 58088

# # # #  dense_1 (Dense)             (None, 9)                 81

# # # # =================================================================
# # # # Total params: 64,854
# # # # Trainable params: 64,854
# # # # Non-trainable params: 0


# #3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['acc'])   # metrics 정확도 확인

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10,   # patience=참을성
#                    verbose=1,   
#                    restore_best_weights=True)

# # ######################### cmp 세이브 파일명 만들기 끗 ###########################

# # import datetime   # 날짜
# # date = datetime.datetime.now()   # 현재 시간
# # print(date)   # 2024-07-26 16:50:13.613311
# # print(type(date))   # <class 'datetime.datetime'>
# # date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
# # print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
# # print(type(date))



# # path = './_save/keras35/'
# # filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# # filepath = "".join([path, 'k35_04',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# # ######################### cmp 세이브 파일명 만들기 끗 ###########################


# # mcp = ModelCheckpoint(
# #     monitor='val_loss',
# #     mode='auto',
# #     verbose=1,
# #     save_best_only=True, # 가장 좋은 놈을 저장
# #     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# # )   # 파일네임, 패스 더하면 요놈

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=5, batch_size=1228,   
#           verbose=1, 
#           validation_split=0.2,
#           callbacks=[es]
# )
# end = time.time()


# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
# print('로스 : ', loss)
# print('acc : ', round(loss[1],2))

# y_predict = model.predict(x_test)




# y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)   # 원핫인코딩한 애들을 다시 원핫인코딩 하기 전으로 변환

# print(y_predict)

# accuracy = accuracy_score(y_test, y_predict)   # 변수에 (초기화 안된) 변수를 넣어서 오류 뜸 
# acc = accuracy_score(y_test, y_predict) # 예측한 y값과 비교
# print("acc_score : ", acc)
# print("걸린 시간 : ", round(end - start,2),'초')


