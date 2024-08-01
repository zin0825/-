# keras26_Scaler01_boaton
# keras28_1_save_model


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
from matplotlib import rc



#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # describe 확인 
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data   # x 데이터 분리   # 스켈링 할 것, x만 (비율만) 건들고 y는 건들면 안됨
y = dataset.target   # y 데이터 분리, sklearn 문법

print(x.shape)   # (506, 13)
print(y.shape)   # (506, )

# x_train과 x_test 하기전에 분리...를 여기보다

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=333)

# 여기서 하는게 더 좋음. 성능이 여기서 정상적으로 잘 작동하고 위는 70%정도로 나옴

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
# 13개의 데이터를 StandardScaler 로 스켈링 한다.
scaler = MinMaxScaler()


scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)   # 위 아래 두 줄을 하나로 줄일 수 있음
x_test = scaler.transform(x_test)   # 변환된 비율만 나오는 것 


print(x_train)   
print(np.min(x_train), np.max(x_train))   
# 0.0 1.0000000000000002 -> 1.0임 파이썬 이진연산을 하기 때문에 오류가 생김. 중요한건 아님 중요한건 스켈링
print(np.min(x_test), np.max(x_test))
# -0.00557837618540494 1.1478180091225068
# 각 컬럼별은 독립적이기에 각 컬럼별로 스켈링이 다르게 함



#2. 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim=13))   # 특성은 많으면 좋음, 한계가 있음, 인풋딤에 다차원 행렬이 들어가면 안됨 
model.add(Dense(10, input_shape=(13,)))   # 이미지 input_shape=(8,8,1) ,하나 있는건 벡터이기 때문   # 13x10=140
model.add(Dense(5))   # 55
model.add(Dense(1))   # 6




# model.save("./_save/keras28/keras28_1_save_model.h5")   # 모델만 저장
# 이 바닥은 h5를 씀. 마지막 /는 거기까지 확장자란 것. 이건 모델파일임

model.summary()   # 모델을 저장할거야




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
          verbose=1, 
          validation_split=0.1
          )
end = time.time()

model.save("./_save/keras28/keras28_3_save_model.h5")   # 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

print("걸린 시간 : ", round(end - start,2),'초')



# 로스 :  97.20042419433594
# r2 score :  0.008957412100013773
# 걸린 시간 :  1.84 초

# 스켈링
# 로스 :  23.58662223815918
# r2 score :  0.7595139881734306
# 걸린 시간 :  2.4 초