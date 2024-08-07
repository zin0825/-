import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape)   # (7, 3) (7,)  / 3차원이라 했는데 x는 2차원이야 아직 rnn이 완벽히 형성되지 않았음

x = x.reshape(x.shape[0], x.shape[1], 1)   # x.reshape(7, 3, 1) 같은거  / 3차원으로 만들어줌
print(x.shape)   # (7, 3, 1)
# 3-D tensor whit shape (batsh_size, timesteps, features)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1)))   # units=60, 3 = timesteps, 1 = features
# model.add(LSTM(units=10, input_shape=(3,1)))   # units=60, 3 = timesteps, 1 = features
model.add(GRU(units=10, input_shape=(3,1)))   # units=60, 3 = timesteps, 1 = features
model.add(Dense(7, activation='relu'))
model.add(Dense(1))   # rnn은 덴스와 바로 연결이 가능하다


model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 475
# Trainable params: 475

# 480개의 비밀을 찾아내랏!!!

# 파라미터 갯수 = units * (units + bias + features)
#              = units*units + units*bias + units*feature 


# - 계산
# model.add(SimpleRNN(3, input_shape=(2,10)))
# 위의 예제의 경우 
# Dh = 3
# t = 2 ( RNN 의 특성상 모든 시점에 히든 스테이트를 공유하므로, time 은 변수의 개수에 관계없다) 
# d = 10 
# 이므로, 아래 계산과정으로 파라미터의 수를 카운팅할 수 있다. 
# # of params = (Dh * Dh) + (Dh * d) + (Dh * 1bias)
# = (3 * 3) + (3 * 10) + (3 * 1) 
# = 42 







# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1500)


# #4. 평가, 예측
# results = model.evaluate(x, y)
# print('loss : ', results)

# x_pred = np.array([8,9,10]).reshape(1, 3, 1)   # [[[8],[9],[10]]]   # 3차원 데이터인데 뒤는 1로 동일하게 맞춰야함
# y_pred = model.predict(x_pred)

# print('[8,9,10]의 결과 : ', y_pred)
# # (3,) -> (1, 3, 1)



# loss :  41.66535186767578
# [8,9,10]의 결과 :  [[1.2831072]]


# SimpleRNN
# loss :  1.2649411473830696e-07
# [8,9,10]의 결과 :  [[10.847718]]

# loss :  4.766136498801643e-06
# [8,9,10]의 결과 :  [[10.909752]]

# loss :  4.222653910562757e-12
# [8,9,10]의 결과 :  [[10.767977]]


# LSTM   # LSTM (시계열을 잡기 위해)잡으려고 트랜스포머가 나왔는데 모든 모델의 베이스가 트랜스포머가 됨
# loss :  0.00037340374547056854   
# [8,9,10]의 결과 :  [[10.885142]]

# loss :  0.0003175536112394184
# [8,9,10]의 결과 :  [[10.855929]]


# GRU
# loss :  0.00021397482487373054
# [8,9,10]의 결과 :  [[10.832041]]

# loss :  7.460675260517746e-05
# [8,9,10]의 결과 :  [[10.90771]]



