from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
# scaler = RobustScaler()

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
                   patience=10, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=10, batch_size=2586, 
                 validation_batch_size=0.3,
                 callbacks=[es])
end = time.time()

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


# 케라스
# acc score :  0.48645485525455234
# 걸린 시간 :  8.08 초
# 로스 :  [0.93593430519104, 0.5576055645942688]

# acc score :  0.6604419813431551
# 걸린 시간 :  7.26 초
# 로스 :  [1.1363941431045532, 0.6739699244499207]

# ra- 3086, ep- 500, ba- 3086
# acc score :  0.8239991738666483
# 걸린 시간 :  483.44 초
# 로스 :  [0.401950865983963, 0.8311246037483215]

# ra- 3086, ep- 700, ba- 3086
# acc score :  0.8350142852225396
# 걸린 시간 :  675.04 초
# 로스 :  [0.37904050946235657, 0.8404185771942139]


# 사이킷런
# acc score :  0.7697497504388834
# 걸린 시간 :  61.56 초
# 로스 :  [0.5086667537689209, 0.7800419926643372]

# acc score :  0.7845168841003752
# 걸린 시간 :  57.34 초
# 로스 :  [0.47943201661109924, 0.7985267043113708]


# 스켈링
# acc score :  0.7627620391724897
# 걸린 시간 :  5.93 초
# 로스 :  [0.519395649433136, 0.777374267578125]


# StandardScaler 
# acc score :  0.8107121957936043
# 걸린 시간 :  5.54 초
# 로스 :  [0.4264499247074127, 0.8238098621368408]


# MaxAbsScaler
# acc score :  0.7599049946645554
# 걸린 시간 :  5.77 초


# RobustScaler
# acc score :  0.8331038518467523
# 걸린 시간 :  5.91 초
# 로스 :  [0.38235512375831604, 0.8430174589157104]

# acc score :  0.8304361295652474
# 걸린 시간 :  5.44 초
# 로스 :  [0.38577184081077576, 0.8414684534072876]