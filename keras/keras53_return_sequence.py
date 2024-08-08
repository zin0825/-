import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time


# rnn 쓰일 일 많음. 주식, 날씨 등등

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 아워너 80

print(x.shape, y.shape)   # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)   # (13, 3, 1)


#2. 모델 구성
model = Sequential()
# model.add(LSTM(60, input_shape=(3,1)))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(1))

model.add(LSTM(60, return_sequences=True, input_shape=(3,1)))   # shape 차원 넣어주기
model.add(LSTM(60))   # 2개 이상이니까 LSTM을 두개 써주고 input_shape는 없애준다
model.add(Dense(60))   # return_sequences 두번째 던져질 땐 시계열이라는 확신이 있을 때. 
model.add(Dense(56))   # 두방 때릴 땐 속도가 현저히 느려진다
model.add(Dense(56))   # 시계열 하나만 넣을 거야 매우 위험한 짓. 일단 해볼것
model.add(Dense(42))
model.add(Dense(42))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))


model.summary()




# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')

# start = time.time()


# np_path = 'c:/프로그램/ai5/_save/keras52_02_summary'
# model.save("c:/프로그램/ai5/_save/keras52_02_summary/k52_02_summary06.h5")


# model.fit(x, y, epochs=1500)


# end = time.time()


# #4. 평가, 예측
# results = model.evaluate(x, y)
# print('loss : ', results)

# x_pred = np.array([50,60,70]).reshape(1, 3, 1)
# y_pred = model.predict(x_pred)

# print('[50,60,70]의 결과 :', y_pred)




# model.add(LSTM(60, input_shape=(3,1)))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(1))
# epochs=1500)
# loss :  0.020263465121388435
# [50,60,70]의 결과 : [[78.79388]]

# loss :  0.0003102876653429121
# [50,60,70]의 결과 : [[79.2954]]

# loss :  5.209277333051432e-06
# [50,60,70]의 결과 : [[79.16645]]
# keras52_02_summaryk42_02_summary0807_1701_0025-1896.7566


# loss :  4.3760410335380584e-05
# [50,60,70]의 결과 : [[79.331665]]
# k52_02_summary02


# @@@@@@
# loss :  9.343335113953799e-05
# [50,60,70]의 결과 : [[79.33196]]  
# k52_02_summary03
# @@@@@@


# epochs=1200
# loss :  1.0061169632535893e-05
# [50,60,70]의 결과 : [[79.04367]]
# k52_02_summary04


# return_sequences=True   # 성능 개 떨어짐
# loss :  492.5444641113281
# [50,60,70]의 결과 : [[[20.398712]
#   [20.397696]
#   [20.380747]]]


# model.add(LSTM(60, return_sequences=True, input_shape=(3,1)))
# model.add(LSTM(60))   # 2개 이상이니까 LSTM을 두개 써주고 input_shape는 없애준다
# loss :  0.005041927099227905
# [50,60,70]의 결과 : [[79.91427]]
# k52_02_summary04

# loss :  0.1425449103116989
# [50,60,70]의 결과 : [[80.8368]]
# k52_02_summary05