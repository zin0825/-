import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import sklearn as sk   # 파이썬 패키지
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston   # 임포트하고

#1. 데이터
dataset = load_boston()   # 여기서 정의
print(dataset) 
print(dataset.DESCR)   # Description 속성을 이용해서 데이터셋의 정보를 확인
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']


x = dataset.data   # skleran 문법 데이터 분리
y = dataset.target   


# R2 0.79 이상 -0.1 (0.69)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.7,
                                                     shuffle=True,
                                                     random_state=104) 

print(x)
print(x.shape)   # (506, 13)  인풋 13
print(y)
print(y.shape)   # (506,) 벡터  아웃풋 1

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(49))
model.add(Dense(80))
model.add(Dense(41))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)



#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2스코어 : ", r2)


# train_size=0.7,
# shuffle=True,
# random_state=104) 
# r2스코어 :  0.6212882612310117

# random_state=200
# r2스코어 :  0.655211685031322

# random_state=300
# r2스코어 :  0.625795016589486

# random_state=1000
# r2스코어 :  0.6169657684490757

# random_state=25
# model.add(Dense(65))
# model.add(Dense(46))
# model.add(Dense(50))
# model.add(Dense(1))
# 로스 :  19.95388412475586
# r2스코어 :  0.6845568378557079