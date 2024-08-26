# m04_1에서 뽑은 4가지 결과로
# 4가지 모델을 맹그러
# input_shape = ()
# 1. 70000,154
# 2. 70000,331
# 3. 70000,486
# 4. 70000,713
# 5. 70000,784 원본

# 시간과 성능을 체크한다.

# 결과 예시 ################
# # 결과1. PCA=154
# 걸린시간 : 000초
# acc = 0.000
# # 결과1. PCA=331
# 걸린시간 : 000초
# acc = 0.000
# # 결과1. PCA=486
# 걸린시간 : 000초
# acc = 0.000
# # 결과1. PCA=713
# 걸린시간 : 000초
# acc = 0.000
# # 결과1. PCA=없음
# 걸린시간 : 000초
# acc = 0.000



from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   #  (2중)분류방식
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)   # (60000,) (10000,)
# DNN에서 60000, 28, 28을 곱했었음
# mnist를 리쉐잎했을 때  28 * 28 = 784
# 4개를 반환해야하는데 나머지는 필요 없을 때 _로 받음
# 전체 데이터의 PCA를 해볼거야 70000만개를. 왜 7만개? 그냥 해볼거야
# 연산을 하는게 아니라 그냥 붙일거야


# 스케일링/ 추가
x_train = x_train/255.
x_test = x_test/255.
# print(np.min(x), np.max(x))   # 0.0 1.0

# exit()


# x = x.reshape(70000, 28*28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
# 쉐이프를 명확히 안다면 좋지만 모르거나 변형을 할 수 있을때 위 방식으로 한다

# print(x_train.shape)   # (60000, 784)

# exit()


n = [154, 331, 486, 713]
print(len(n))   # len 길이

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    
    #2. 모델
    model = Sequential()
    model.add(Dense(128, input_shape=(n[i],)))   # 흑백 1은 묵음
    model.add(Dense(128, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(44, activation='relu'))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(10, activation='softmax'))


    #3. 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    start = time.time()

    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=5, verbose=1,
                    restore_best_weights=True)

    hist = model.fit(x_train1, y_train, epochs=15, batch_size=188,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[es]) 

    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)

    y_pred = model.predict(x_test1)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
    y_pred = np.argmax(y_test, axis=1).reshape(-1,1)

    print('결과 PCA :', n[i] )
    print('걸린 시간 : ', round(end - start,2), "초")
    print('acc : ', loss[1])   # 0은 로스, 1은 acc



# 결과 PCA : 154
# 걸린 시간 :  8.43 초
# acc :  0.9721999764442444
# 결과 PCA : 331
# 걸린 시간 :  8.52 초
# acc :  0.9674999713897705
# 결과 PCA : 486
# 걸린 시간 :  7.79 초
# acc :  0.963699996471405
# 결과 PCA : 713
# 걸린 시간 :  9.42 초
# acc :  0.9677000045776367



# #2. 모델
# model = Sequential()
# model.add(Dense(128, input_shape=(28*28,)))   # 흑백 1은 묵음
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(96, activation='relu'))
# model.add(Dense(96, activation='relu'))
# model.add(Dense(44, activation='relu'))
# model.add(Dense(22, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# # exit()


# #3. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['acc'])

# start = time.time()

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)

# hist = model.fit(x_train, y_train, epochs=15, batch_size=188,
#                  verbose=1,
#                  validation_split=0.2,
#                  callbacks=[es]) 

# end = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# # print('acc : ', round(loss[1],2))

# y_pred = model.predict(x_test)

# y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
# y_pred = np.argmax(y_test, axis=1).reshape(-1,1)
# print(y_pred)

# acc = accuracy_score(y_test, y_pred)
# print('acc : ', acc)
# print('걸린 시간 : ', round(end - start,2), "초")
# print('결과 : ', y_pred)




















