# import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time




# a = np.array([[1,2,3,4,5,6,7,8,9,10],
#              [9,8,7,6,5,4,3,2,1,0]]).reshape(10,2)
# print(a.shape)   # (10, 2)

# size = 5
# print(a)
# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)
#     return np.array(aaa)

# bbb = split_x(a, size)
# print(bbb)
# # [[[ 1  2]
# #   [ 3  4]
# #   [ 5  6]
# #   [ 7  8]
# #   [ 9 10]]

# #  [[ 3  4]
# #   [ 5  6]
# #   [ 7  8]
# #   [ 9 10]
# #   [ 9  8]]

# #  [[ 5  6]
# #   [ 7  8]
# #   [ 9 10]
# #   [ 9  8]
# #   [ 7  6]]

# #  [[ 7  8]
# #   [ 9 10]
# #   [ 9  8]
# #   [ 7  6]
# #   [ 5  4]]

# #  [[ 9 10]
# #   [ 9  8]
# #   [ 7  6]
# #   [ 5  4]
# #   [ 3  2]]

# #  [[ 9  8]
# #   [ 7  6]
# #   [ 5  4]
# #   [ 3  2]
# #   [ 1  0]]]
# print(bbb.shape)   # (6, 5, 2)

# x = bbb[:, :-1]
# y = bbb[:, -1]  
# print(x, y)
# # [[[ 1  2]
# #   [ 3  4]
# #   [ 5  6]
# #   [ 7  8]]

# #  [[ 3  4]
# #   [ 5  6]
# #   [ 7  8]
# #   [ 9 10]]

# #  [[ 5  6]
# #   [ 7  8]
# #   [ 9 10]
# #   [ 9  8]]

# #  [[ 7  8]
# #   [ 9 10]
# #   [ 9  8]
# #   [ 7  6]]

# #  [[ 9 10]
# #   [ 9  8]
# #   [ 7  6]
# #   [ 5  4]]

# #  [[ 9  8]
# #   [ 7  6]
# #   [ 5  4]
# #   [ 3  2]]] [[ 9 10]
# #  [ 9  8]
# #  [ 7  6]
# #  [ 5  4]
# #  [ 3  2]
# #  [ 1  0]]
# print(x.shape, y.shape)   # (6, 4, 2) (6, 2)


import numpy as np
a = np.array([[1,2,3,4,5,6,7,8,9,10],
             [9,8,7,6,5,4,3,2,1,0]]).T
print(a.shape)   # (10, 2)

size = 6

print(len(a))   # 10

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   # (5, 6, 2)


x = bbb[:, :-1]   # 
y = bbb[:, -1, 0]   # 
print(x, y)
print(x.shape, y.shape)   # 

# [[5 5]
#   [6 4]
#   [7 3]
#   [8 2]
#   [9 1]]] [ 6  7  8  9 10]
# (5, 5, 2) (5,)

#2. 모델 구성
model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape=(5,2)))   # shape 차원 넣어주기
model.add(LSTM(60)) 
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



np_path = 'c:/프로그램/ai5/_save/keras54_02'
model.save("c:/프로그램/ai5/_save/keras54_02/k54_02_.h5")


# model.fit(x, y, epochs=1500)


end = time.time()


#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

y_pred = model.predict([[[6,4],[7,3],[8,1],[9,1],[10,0]]])
# y_pred = model.predict(x_pred)

print('[11]의 결과 :', y_pred)


# loss :  3.6379788613018216e-13
# [11]의 결과 : [[10.419566]]


# loss :  1.5111982776261357e-08
# [11]의 결과 : [[10.575104]]
# keras54_02/k54_02_.h5













