from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (581012, 54) (581012,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))


# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (581012, 8)

y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (581012, 7)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)   # y는 예스 아니고 y

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# print(pd.value_counts(y_train))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0



#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=54))
model.add(Dense(90))
model.add(Dense(46))
model.add(Dense(6))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras30_mcp/keras30_10_fetch_covtpe.hdf5')

hist = model.fit(x_train, y_train, epochs=10, batch_size=2586, 
                 validation_split=0.3,
                 callbacks=[es, mcp])
end = time.time()

model.save('./_save/keras30_mcp/keras30_10_fetch_covtpe_save.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('ACC : ', round(loss[1], 3))

y_pred = model.predict(x_test)
print(y_pred[:20])
y_pred = np.round(y_pred)
print(y_pred[:20])

accuracy_score = accuracy_score(y_test, y_pred)
print(y_pred)

print('acc score : ', accuracy_score)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)


# acc score :  0.7576159168359092
# 걸린 시간 :  5.82 초
# 로스 :  [0.5332769155502319, 0.7746893167495728]