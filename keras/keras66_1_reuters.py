from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,   # 1000이든 10000이든 상관없음 
    # maxlen=10,   # 100개까지 있는것만 해라 / 패드 시퀀서하면서 잘라버리면 됨
    test_split=0.2,
)

print(x_train)
print(x_train.shape, x_test.shape)   # (8982,) (2246,)
print(y_train.shape, y_test.shape)   # (8982,) (2246,)
print(y_train)   # [ 3  4  3 ... 25  3 25] y값이 다중분류란 것
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train)))   # 46
# 텐서에서 준거니까 당연히 넘파이임
# 중간에 보면 list가 있음 /  넘파이 안에 리스트가 있음 
# 리스트의 문제가 데이터의 길이가 어떤건 30 어떤건 60개 일정하지 않음 -> 조절해야함

print(type(x_train))   # <class 'numpy.ndarray'>
print(type(x_train[0]))   # 첫번째의 타입을 보자 / <class 'list'> -> 넘파이로 변경해야함
print(len(x_train[0]), len(x_train[1]))   # 87 56

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))   # 2376 / 파이썬 함수
# 리스트 안에 있으니까 다 찾아 봐야함 -> for 문
# for i에 들어가는거 87-> len(i)-> 87
print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train))   # 13 / 파이썬 함수
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))   
#  145.5398574927633 / 넘파이
# 위에 한건 얼마큼 자를건가/ 인풋렝스를 조지기 위해

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
# padding='pre', post해도 상관없음. 돌려서 성능 좋은걸로 선택
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')

# 투 카테고리든 캣더미든 상관 없음
# y 원핫하고
# 맹그러봐!!! 

from tensorflow.keras.utils import to_categorical


y_train = to_categorical(y_train)   # y 원핫
y_test = to_categorical(y_test)

print(y_train.shape)   # (8982, 46)
print(y_test.shape)   # (2246, 46)

# exit()



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=66)


scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100))   # 여기가 x 원핫
# model.add(Conv1D(50, 3))
# model.add(Conv1D(50, 3))
# model.add(BatchNormalization())
model.add(LSTM(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(46))
model.add(Dense(46))
model.add(Dense(46))
model.add(Dense(46, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=288, validation_split=0.2)


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results)


# loss :  [3.3585503101348877, 0.6678539514541626]