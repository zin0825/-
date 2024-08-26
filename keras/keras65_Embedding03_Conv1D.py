# 땡겨서 맹그러!!!


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#1. 데이터

docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', ' 한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다.', '태운이 또 구라친다.',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}


x = token.texts_to_sequences(docs)
print(x)
print(type(x))   # <class 'list'>
# [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], [10, 11, 12, 13, 14], [15], 
# [16], [17, 18], [19, 20], 
# [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]


from tensorflow.keras.preprocessing.sequence import pad_sequences
# 맹그러봐!!! (15,5)


pad_x = pad_sequences(x, 
                    #   padding = 'pre' / 디폴트, 'post',
                      maxlen=5,   # 최대길이에 맞추려고 함
                    #   truncating='pre')   # pre 앞에서 자른다, post 뒤에서 자른다
                    )
print(pad_x)
print(pad_x.shape)   # (15, 5)

# 원래는 원핫을 해야하는데 그런거 생각하지 말고 모델을 만들자

#2. 모델
################### DNN 맹그러봐 ###################
x_pred = ['태운이는 참 재미없다.']

x_token = Tokenizer()
x_token.fit_on_texts(x_pred)
print(x_token.word_index)
# {'태운이는': 1, '참': 2, '재미없다': 3}

x_pred2 = token.texts_to_sequences(x_pred)
print(x_pred2)   # [[1, 22]]

x_pred2 = pad_sequences(x_pred2, maxlen=5)
print(x_pred2)   # [[ 0  0  0  1 22]]

# x = x/255.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(pad_x, labels, train_size=0.8, random_state=66)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=80, kernel_size=2, input_shape=(5, 1)))
model.add(Conv1D(80, 2))
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# model.add(LSTM(80, input_shape=(5, 1)))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)


start = time.time()

model.fit(pad_x, labels, epochs=100, batch_size=32,
                 verbose=1,
                 callbacks=[es])

end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_pred2)

# print('acc : ', round(loss,2))
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss[0])
print('태운이는 참 재미없다. : ', np.round(y_pred))


# DNN
# 걸린 시간 :  1.2 초
# 로스 :  0.6396433115005493
# 태운이는 참 재미없다. :  [[0.]]

# 걸린 시간 :  1.2 초
# 로스 :  0.4924970865249634
# 태운이는 참 재미있다. :  [[1.]]


# LSTM
# 걸린 시간 :  2.32 초
# 로스 :  0.0027294419705867767
# 태운이는 참 재미없다. :  [[0.]]

# 걸린 시간 :  2.37 초
# 로스 :  0.08415871858596802
# 태운이는 참 재미있다. :  [[1.]]


# Conv1D
# 걸린 시간 :  2.61 초
# 로스 :  0.5482007265090942
# 태운이는 참 재미없다. :  [[0.]]

# 걸린 시간 :  4.09 초
# 로스 :  0.5180230736732483
# 태운이는 참 재미있다. :  [[1.]]
















