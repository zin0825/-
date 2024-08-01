from sklearn.datasets import load_digits   # digits 숫자
import pandas as pd

x, y = load_digits(return_X_y=True)   # x와 y로 바로 반환해줌
print(x)
print(y)
print(x.shape, y.shape)   # (1797, 64) (1797,)   이미지는 0에서 225의 숫자를 부여함 225가 가장 진함 놈
# 1797장의 이미지가 있는데 8바이8 짜리를 64장으로 쭉 한것, 원래는 (1797,8,8)의 이미지 인데 칼라는 (1797,8,8,1)

print(pd.value_counts(y, sort=False))   # 확인, ascending=True 오름차순  # y라벨 10개
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
# y = to_categorical(y)   # 케라스
# print(y)
# print(y.shape)   # (1797, 10)


y = pd.get_dummies(y)   # 판다스
print(y)
print(y.shape)   # (1797, 10)

# y = y.reshape(-1, 1)   # 사이킷런
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape)   # (1797, 10)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)

print(x.shape, y.shape)   # (1797, 64) (1797, 10)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)   

print(x_train)   
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0666666666666667



#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=64))
model.add(Dense(90))
model.add(Dense(46))
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=180, batch_size=3086, 
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


# # 케라스
# # acc score :  0.9777777777777777
# # 걸린 시간 :  0.87 초
# # 로스 :  [0.11199933290481567, 0.9833333492279053]

# # 판다스
# # acc score :  0.9333333333333333
# # 걸린 시간 :  0.88 초
# # 로스 :  [0.18980160355567932, 0.9388889074325562]

# # 사이킷런
# # acc score :  0.9833333333333333
# # 걸린 시간 :  0.8 초
# # 로스 :  [0.0742197334766388, 0.9833333492279053]

# acc score :  0.9888888888888889
# 걸린 시간 :  1.25 초
# 로스 :  [0.05170276015996933, 0.9888888597488403]


# 스켈링
# acc score :  0.9555555555555556
# 걸린 시간 :  1.14 초
# 로스 :  [0.14160378277301788, 0.9611111283302307]


# StandardScaler 
# acc score :  0.9611111111111111
# 걸린 시간 :  1.06 초
# 로스 :  [0.15787474811077118, 0.9611111283302307]


# MaxAbsScaler
# acc score :  0.95
# 걸린 시간 :  1.2 초
# 로스 :  [0.19487249851226807, 0.949999988079071]

# acc score :  0.9666666666666667
# 걸린 시간 :  1.16 초
# 로스 :  [0.12018055468797684, 0.9777777791023254]


# RobustScaler
# acc score :  0.9833333333333333
# 걸린 시간 :  1.09 초
# 로스 :  [0.1139720156788826, 0.9833333492279053]