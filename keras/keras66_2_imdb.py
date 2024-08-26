from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,
                                                    #   maxlen=10,
                                                    #   test_split=0.2
                                                    )


print(x_train)
print(x_train.shape, x_test.shape)   # (25000,) (25000,)
print(y_train.shape, y_test.shape)   # (25000,) (25000,)
print(y_train)   # [1 0 0 ... 0 1 0]
print(np.unique(y_train))   # [0 1]
print(len(np.unique(y_train)))   # 2
# # 텐서에서 준거니까 당연히 넘파이임
# # 중간에 보면 list가 있음 /  넘파이 안에 리스트가 있음 
# # 리스트의 문제가 데이터의 길이가 어떤건 30 어떤건 60개 일정하지 않음 -> 조절해야함

print(type(x_train))   # <class 'numpy.ndarray'>
print(type(x_train[0]))   # 첫번째의 타입을 보자 / <class 'list'> -> 넘파이로 변경해야함
print(len(x_train[0]), len(x_train[1]))   # 218 189

print("imdb : ", max(len(i) for i in x_train))   # 2494
# # 리스트 안에 있으니까 다 찾아 봐야함 -> for 문
# # for i에 들어가는거 87-> len(i)-> 87
print("imdb : ", min(len(i) for i in x_train))   # 11
print("imdb : ", sum(map(len, x_train)) / len(x_train))   
# #  238.71364
# # 위에 한건 얼마큼 자를건가/ 인풋렝스를 조지기 위해


# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
# padding='pre', post해도 상관없음. 돌려서 성능 좋은걸로 선택
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')


# y 원핫하고
# 맹그러봐!!! 


# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)   # y 원핫
# y_test = to_categorical(y_test)

print(y_train.shape)   # (25000,)
print(y_test.shape)   # (25000,)


# exit()


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=66)


scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



#3. 컴파일, 훈력
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=288, validation_split=0.2)



#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results)


# y 원핫 ㅇ
# loss :  [0.6934144496917725, 0.5192400217056274]

# y 원핫 x
# StandardScaler
# loss :  [0.9672138094902039, 0.5089600086212158]

# MinMaxScaler
# loss :  [0.6923673152923584, 0.5023999810218811]

# MaxAbsScaler
# loss :  [0.692287266254425, 0.5023999810218811]

# RobustScaler
# loss :  [1.0200226306915283, 0.5238000154495239]
