import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.utils import to_categorical
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
# [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]


from tensorflow.keras.preprocessing.sequence import pad_sequences
# 맹그러봐!!! (15,5)


pad_x = pad_sequences(x, 
                    padding = 'pre',   # pre  앞에 0 디폴트, post 뒤에 0 / 문장의 길이가 다 달라서 긴 문장에 맞추기 위해 앞에 0을 채운다
                      maxlen=5,   # 최대길이에 맞추려고 함
                    #   truncating='pre')   # pre 앞에서 자른다 디폴트, post 뒤에서 자른다
                    )
print(pad_x)
print(pad_x.shape)   # (15, 5) -> (15, 5, 30) => x_train 5, 30 실질적인 인풋쉐이프   


# pad_x = to_categorical(pad_x)
# print(pad_x.shape)   # (15, 5, 31)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# 데이터가 많을 때 원핫인코딩을 하면 더 많아진다
# 앞에서는 원핫인코딩을 하고 모델을 했는데 임베딩은 그럴 필요 없다
# 모델에서 임베딩을 쉐이프 해준다
# 원핫 하지 않은 데이터를 넣어준다 (15, 5) 
# 패딩까진 되어있어야함 가로 세로 맞추기 위해
# (15, 5) 말 뭉치의 갯수는 31개 였다

# 원핫인코딩을 하는 이유는 인덱스들이 혼합되면서 혼동되는 문제를 해결한다. 
# 하지만, 데이터의 수가 늘어나면 인덱스가 기하급수적으로 늘어나는 문제가 발생한다.
# 또한, 0의 갯수가 엄청 늘어나면서 연산량도 늘어나고, 결과값이 0으로 수렴한다.
# 이러한 문제를 해결하기 위해서 의미가 유사한 것을 기반으로 군집화할 수 있다. = Embedding


x_pred = ['태운이 참 재미없다']

token.fit_on_texts(x_pred)
print(token.word_index)
# {'태운이는': 1, '참': 2, '재미없다': 3}

x_pred2 = token.texts_to_sequences(x_pred)
# print(x_pred2)   # [[1, 22]]

x_pred2 = pad_sequences(x_pred2, 5)
print(x_pred2)   # [[0 0 4 1 3]]
print(x_pred2.shape)   # (1, 5)

model = Sequential()
############################# 임베딩 1 ############################# / 정석
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))   # (None, 5, 100) 
# # 임베딩 레이어만 첫번째가 인풋이 아닌 아웃풋이 됨
# # 인풋딤 31 = 단어 사전의 개수 (조절 가능 함) / 갯수를 뽑아 줘야 함
# # 아웃풋 딤 = 다음 레이어에 전달하는 실질적인 노드 갯수
# # 인풋 렝스 5 = 전처리한 패딩 (정제)된 데이터의 가로 길이

# model.add(LSTM(10))   # (None, 10) 
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))   

# model.summary()

# output_dim=100
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

# =================================================================
# Total params: 3,100
# Trainable params: 3,100
# Non-trainable params: 0
# 아웃풋의 개수만큼 곱하기 인풋의 개수
# 임베딩 시계열데이터에서 사용 가능 2차원
# 나온거 보면 3차원 LSTM과 연결 가능. 통상 자연어처리, 시계열 처리에서 많이 사용
# 단어 사전의 개수, 패딩의 가로 길이만 알아두면 됨
# 인풋 랭스는 바꿀 수 없음 / 데이터가 15, 5이기 때문에 5로 고정 된것 / 행무시, 열우선 그래서 행은 5로 고정

####################################################################

############################# 임베딩 2 ############################# / 렝스 없음
# model.add(Embedding(input_dim=31, output_dim=100))

# model.add(LSTM(10))   # (None, 10) 
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))   


# model.summary()

# model.add(Embedding(input_dim=31, output_dim=100))
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100   # 렝스 없어도 알아서 조절해서 맞춰준다

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
####################################################################

############################# 임베딩 3 ############################# / 인풋딤 변경
# model.add(Embedding(input_dim=100, output_dim=100))   
# 인풋딤 바꿔도 돌아감. 아무 숫자넣어도 돌아가지만 성능 차이 남.
# 갯수는 가급적이면 단어 사전에 맞춰서 넣어주는게 좋음
# 100개를 했을 경우 단어 사정보다 훨씬 많으면 과소비 (공간낭비) 됨 / 백원 낼거 천원 냄
# 랭스도 가급적 명시하는게 좋음. 행무시, 열우선이니까 열의 값에 맞춰줘야함
# 원핫보다 간결하고 좋음. 압축형식이라서 많은 데이터를 2차원 데이터로 만들어 주기에 효율적임

# input_dim=30   # 디폴트
# input_dim=20   # 단어 사전의 갯수보다 작을 때 : 연산량 줄어, 단어사전에서 임의로 빼 : 성능 조금 저하
# input_dim=40   # 단어 사전의 갯수보다 클 때 : 연산량 늘어, 임의의 랜덤 임베딩 생성 : 성능 조금 저하

# model.add(LSTM(10))   # (None, 10) 
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))   
####################################################################

############################# 임베딩 4 ############################# / 인풋 명시 x
model.add(Embedding(31, 100))   # 잘 돌아감
# model.add(Embedding(31, 100, 5))  
# ValueError: Could not interpret initializer identifier: 5   # 에러. 5개 뭔지 모른다
# model.add(Embedding(31, 100, input_length=5))   # 잘 돌아가
# model.add(Embedding(31, 100, input_length=1))   # 잘 돌아가 / 약수를 맞출 필요는 없다
# inout_lenght 1, 5는 돼. 2,3,4,6... 안돼
# 시계열 데이터는 일부러 3차원 데이터 만들어 던졌는데 임베딩은 크게 2차원으로 만들어서 던진다

model.add(LSTM(10))   # (None, 10) 
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))   


# model.summary()

# model.add(Embedding(31, 100)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
####################################################################



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)


#4. 평가, 예측
results = model.evaluate(pad_x, labels)
print('loss : ', results)

y_pred = model.predict(x_pred2)
print('태운이 참 재미없다 : ', np.round(y_pred))


# loss :  [0.0260381568223238, 1.0]

# loss :  [0.04608897492289543, 1.0]
# 태운이 참 재미없다 :  [[1.]]