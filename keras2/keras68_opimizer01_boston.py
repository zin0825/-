from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337)   # 값 고정
np.random.seed(337)


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,
                                                    random_state=337,
                                                    )

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam   # ** 중요
# 러닝 레이트는 훈련의 비율을 조정해주는 것

# learning_rate = 0.001   # r2 : 0.6366904421717885

# learning_rate = 0.01   # r2 : 0.6304700624761163

# learning_rate = 0.0001   # r2 : -2.7350779742844455

# learning_rate = 0.002   # r2 : 0.6364639071767033

# learning_rate = 0.005   # r2 : 0.6334064705081626

# learning_rate = 0.0015   # r2 : 0.6359603503657396

# learning_rate = 0.0008   # r2 : 0.6373160244542878

# learning_rate = 0.0007   # r2 : 0.6375050183176865

learning_rate = 0.0007
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))   
# learning_rate=learning_rate 숫자 장난. 0.01로 넣어줘도 됨
# learning_rate 디폴트 0.001

model.fit(x_train, y_train, 
          validation_split=0.2,
          epochs=100,
          batch_size=32,
          )


#4. 평가, 예측
print('=================1. 기본 출력 =================')
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


# 로스 : 34.11126708984375
# r2 : 0.6367126079664703

# import tensorflow as tf
# tf.random.set_seed(337)   # 값 고정
# 로스 : 34.11334991455078
# r2 : 0.6366904421717885

# learning_rate = 0.001
# 로스 : 34.11334991455078
# r2 : 0.6366904421717885

# learning_rate = 0.01
# 로스 : 34.697418212890625
# r2 : 0.6304700624761163

# learning_rate = 0.0001
# 로스 : 350.7092590332031
# r2 : -2.7350779742844455

# learning_rate = 0.002
# 로스 : 34.134620666503906
# r2 : 0.6364639071767033

# learning_rate = 0.005
# 로스 : 34.42170333862305
# r2 : 0.6334064705081626

# learning_rate = 0.0015
# 로스 : 34.18190383911133
# r2 : 0.6359603503657396

# learning_rate = 0.0008
# 로스 : 34.05460739135742
# r2 : 0.6373160244542878

# learning_rate = 0.0007
# 로스 : 34.036861419677734
# r2 : 0.6375050183176865

# print('=================1. 기본 출력 =================')
# loss = model.evaluate(x_test, y_test, verbose=0)
# print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 한번에 출력

# y_predict = model.predict(x_test, verbose=0)
# r2 = r2_score(y_test, y_predict)
# print('lr : {0}, r2 : {1}'.format(learning_rate, r2))
# =================1. 기본 출력 =================
# lr : 0.0007, 로스 : 34.036869049072266
# lr : 0.0007, r2 : 0.6375049830314161


#################### [ 실습 ] ####################
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]