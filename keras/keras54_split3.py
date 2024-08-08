import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time



a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))   # 102부터 107을 찾아라

# 맹그러봐!!!

print(a.shape)   # (100,)
print(x_predict.shape)   #(10,)

size = 11   # **내가 구하고자 하는 숫자에 1을 더해준다. 함수 스플릿을 거쳐서 나오는 데이터의 컬럼 수와 같다

print(len(a))   # 100

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   # (90, 11)


x = bbb[:, :-1]   # **의 결과는 여기서 
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)   # (90, 10) (90,)


#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(10, 1), activation='relu')) 
# model.add(LSTM(60, return_sequences=True, input_shape=(90,10)))   # shape 차원 넣어주기
# model.add(LSTM(60))
model.add(Dense(60, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))




#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=16,
          verbose=1)

start = time.time()

np_path = 'c:/프로그램/ai5/_save/keras54_03'
model.save("c:/프로그램/ai5/_save/keras54_03/k54_01_.h5")


end = time.time()


#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array(range(96, 106)).reshape(1, 10, 1)  

y_pred = model.predict(x_predict)


# y_pred = model.predict(np.array(range(96, 106)))
# y_pred = model.predict(x_pred)

print('[11]의 결과 :', x_predict)


# x_predict = np.array(range(96, 106)).reshape(1, 10, 1)  # 이것만 했을 때
# loss :  0.00012529946980066597
# [11]의 결과 : [[[ 96]
#   [ 97]
#   [ 98]
#   [ 99]
#   [100]
#   [101]
#   [102]
#   [103]
#   [104]
#   [105]]]


