# 15개의 행에서
# 5개를 더 넣어서 (임의로) (2행은 6개 이상 단어)
# 예) 반장 주말에 출근 혜지 안혜지 안혜지 //0

# 맹그러
# 5개 조합해서 프레딕트를 만들어

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
    '반장 주말에 출근 혜지 안혜지 안혜지', '현아 쾌할하지', '누리 손목 아프지',
    '혜지 가볍게 출근하지', '진영이 4090 질러서 이제 돈이 없지'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '반장': 3, 
# '혜지': 4, '안혜지': 5, '재미있다': 6, 
# '최고에요': 7, '잘만든': 8, '영화예요': 9, 
# '추천하고': 10, '싶은': 11, '영화입니다': 12, 
# '한': 13, '번': 14, '더': 15, 
# '보고': 16, '싶어요': 17, '글쎄': 18, 
# '별로에요': 19, '생각보다': 20, '지루해요': 21, 
# '연기가': 22, '어색해요': 23, '재미없어요': 24, 
# '재미없다': 25, '재밋네요': 26, '준영이': 27, 
# '바보': 28, '잘생겼다': 29, '태운이': 30, 
# '또': 31, '구라친다': 32, '주말에': 33, 
# '출근': 34, '현아': 35, '쾌할하지': 36, 
# '누리': 37, '손목': 38, '아프지': 39, 
# '가볍게': 40, '출근하지': 41, '진영이': 42, 
# '4090': 43, '질러서': 44, '이제': 45, 
# '돈이': 46, '없지': 47} 

x = token.texts_to_sequences(docs)
# print(x)
# print(type(x))   # <class 'list'>
# [[2, 6], [1, 7], [1, 8, 9], 
# [10, 11, 12], [13, 14, 15, 16, 17], [18], 
# [19], [20, 21], [22, 23], 
# [24], [2, 25], [1, 26], 
# [27, 28], [3, 29], [30, 31, 32], 
# [3, 33, 34, 4, 5, 5], [35, 36], [37, 38, 39], 
# [4, 40, 41], [42, 43, 44, 45, 46, 47]]


# exit()

from tensorflow.keras.preprocessing.sequence import pad_sequences



pad_x = pad_sequences(x, 
                    padding = 'pre',   # 디폴트, 'post',
                      maxlen=5,   # 최대길이에 맞추려고 함
                    #   truncating='pre')   # pre 앞에서 자른다, post 뒤에서 자른다
                    )
# print(pad_x)
# print(pad_x.shape)   # (20, 5)   


pad_x = to_categorical(pad_x)
print(pad_x.shape)   # (20, 5, 48)

# exit()



#2. 모델
################### DNN 맹그러봐 ###################
x_pred = ['태운이 참 재미없다']


token.fit_on_texts(x_pred)
print(token.word_index)
# {'태운이는': 1, '참': 2, '재미없다': 3}

x_pred2 = token.texts_to_sequences(x_pred)
print(x_pred2)   # [[5, 1, 3]]

x_pred2 = pad_sequences(x_pred2, 5)
print(x_pred2)   # [[0 0 5 1 3]]
print(x_pred2.shape)   # (1, 5)

x_pred2 = to_categorical(x_pred2, num_classes=48)
print(pad_x.shape)   # (20, 5, 48)
print(x_pred2.shape)   # (1, 5, 48)


x_pred3 = x_pred2.reshape(1, 5, 48)
print(x_pred3.shape)   # (1, 5, 48)

# exit()
# exit()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(pad_x, labels, 
                                                    train_size=0.8, 
                                                    random_state=66)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=80, kernel_size=2, input_shape=(5, 48)))
model.add(Conv1D(80, 2))
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# model.add(LSTM(80, input_shape=(5, 31)))
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
print('태운이 참 재미없다 : ', np.round(y_pred))


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


# ohe_LSTM
# 걸린 시간 :  2.32 초
# 로스 :  1.0291292710462585e-05
# 태운이 참 재미없다. :  [[0.]]

# 걸린 시간 :  2.32 초
# 로스 :  1.7709187005721105e-08
# 태운이 참 재미있다. :  [[1.]]


# ohe_Conv1D
# 걸린 시간 :  2.64 초
# 로스 :  1.3095663575768413e-07
# 태운이 참 재미없다 :  [[1.]]

# 걸린 시간 :  2.62 초
# 로스 :  1.7674284435997833e-07
# 태운이 참 재미있다 :  [[1.]]


# ohe_과제 
# 걸린 시간 :  4.0 초
# 로스 :  1.8330156308365986e-06
# 태운이 참 재미없다 :  [[1.]]












