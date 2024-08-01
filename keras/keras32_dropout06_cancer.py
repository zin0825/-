import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_breast_cancer()
# print(datasets)   # 한개의 행[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01, 1.189e-01]
#  y 값 array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...

print(datasets.DESCR)   
# Number of Instances: 569 행
# Attributes: 30  속성, 열
# Missing Attribute Values: None 결측치 있냐
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (569, 30) (569,)

# y는 넘파이 데이터에서 암이 반절이 걸렸다. 0,1,2, 다중 분류일 경우 다 확인해야하는데
# y의 라벨값을 묻는것이 있다.
# 넘파이에서 y가 0과 1의 종류

print(type(x))   #<class 'numpy.ndarray'>

"""np.unique(datasets)
unique, counts= np.unique(datasets, return_counts=True)
print(unique, counts)
dict(Zip(unique, counts))
"""

# 0과 1의 갯수가 몇개인지 찾아요.
print(np.unique(y, return_counts=True))   
# (array([0, 1]), array([212, 357], dtype=int64))

# print(y.value_counts())   에러
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=99, 
                                                    train_size=0.9)

print(x_train.shape, y_train.shape)   # (398, 30) (398,)
print(x_test.shape, y_test.shape)   # (171, 30) (171,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))   # -0.02318339100346023 1.317775390162621



#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=30))   
model.add(Dropout(0.4)) 
model.add(Dense(16,activation='relu'))   # 통상적으로 2의 배수, 이진법이 잘 먹힘
model.add(Dropout(0.4)) 
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(16, activation='sigmoid'))   # 중간에 시그모이드 써도 됨
model.add(Dropout(0.3)) 
model.add(Dense(16, activation='relu'))   # 렐루, 시그모이드 뭐가 좋아? 해봐야함
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 분류에서  마지막 레이어 엑큐베이션은 리니어,
# 이진 분류에서는 마지막 엑티베이션은 시그마 = 0에서 1사이의 값을 바꿔준다

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])   #  accuracy, mse
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',   # 최소값을 찾을거야
    mode = 'min',   # 모르면 auto 알고있으니가 min
    patience=20,
    verbose=1,
    restore_best_weights=True,   # y=wx + b의 최종 가중치 어쩔 땐 안쓰는게 좋을 수도 있음
)   # 얼리스탑핑을 정리하는게 끝

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
filepath = "".join([path, 'k32_06',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
) 
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 8, 
                 verbose=1, validation_split=0.2,
                 callbacks=[es, mcp]
                 )   # [] = 리스트 / 두개이상은 리스트 = 나중에 또 친구 나오겠다
end = time.time()



# 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1) 
print("로스 : ", loss[0])   # 로스, 컴파일의 metrics=['acc']를 로스값에 넣어서..
print("ACC : ", round(loss[1], 3))   # round() 소수점 여기는 메트릭스


y_pred = model.predict(x_test)   # y_predict 값이  [0.98425215] 등으로 표시
# print(y_predict[:10])
print(y_pred[:20])
y_pred = np.round(y_pred)   # y_pred 의 소속은 넘파이 0이나 1로 나와야해서 반올림을 해줬다  [1.]으로 표시
print(y_pred[:20])

from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)   
# 0.0033등으로 나와야 하는데 멋대로 전부 1이 들어가버림 (에러) / 0, 1이라는 정수를 올려줘, 비교해줘 
print("acc_score : ",  accuracy_score)
print("걸리시간 : ", round(end - start, 2), "초")
print("로스 : ", loss[0])

r2 = r2_score(y_test, y_pred)
print('r2 score : ', r2)


# acc_score :  0.9649122807017544
# 걸리시간 :  3.25 초
# 로스 :  0.025403909385204315


# Dropout (0.3, 0.3, 0.3)
# acc_score :  0.9649122807017544
# 걸리시간 :  3.88 초
# 로스 :  0.03011304698884487
# r2 score :  0.8323529411764705

# (0.4, 0.4, 0.3, 0.3)
# acc_score :  0.9473684210526315
# 걸리시간 :  4.79 초
# 로스 :  0.03343532234430313
# r2 score :  0.7485294117647059