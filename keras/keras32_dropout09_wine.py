from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

from tensorflow.keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)

print(x.shape)
print(y.shape)
# (178, 13)
# (178,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.05128205128205132 0.9896551724137932





#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=13))
model.add(Dropout(0.4)) 
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.4)) 
model.add(Dense(46, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(22, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, 
                   verbose=1,
                   restore_best_weights=True)
 
 ######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   # 현재 시간
print(date)   # 2024-07-26 16:50:13.613311
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
print(type(date))



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_09',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################
 
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)
 
 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("ACC : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print(y_predict)


print("acc score : ", accuracy_score)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
print("걸린 시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


# acc score :  0.9444444444444444
# 걸린 시간 :  1.92 초
# 로스 :  [0.08138372004032135, 0.9444444179534912]


# Dropout
# acc score :  0.9444444444444444
# 걸린 시간 :  2.12 초
# 로스 :  [0.15466207265853882, 0.9444444179534912]
