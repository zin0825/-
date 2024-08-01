









# [실습]
import numpy as np
from keras.models import Sequential, Model   # 시퀀셜은 시퀀셜 하나만이지만 모델은 모멜, 인풋이 있어야함
from keras.layers import Dense, Input, Dropout

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0],
]
)   #  예측 가능한 데이터

print(x.shape)

x = x.T
print(x.shape)

y = np.array([1,2,3,4,5,6,7,8,9,10])


# #2-1. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(10, input_dim = (3,)))
# model.add(Dense(9))
# model.add(Dropout(0.3))
# model.add(Dense(8))
# model.add(Dropout(0.2))
# model.add(Dense(7))
# model.add(Dense(1))

#2-2. 모델구성(함수형)   # 성능차이는 없음, 표현방식이 다를 뿐
input1 = Input(shape=(3,))   # = input_dim = (3,)))
dense1 = Dense(10, name='ys1')(input1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
dense2 = Dense(9, name='ys2')(dense1) 
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8, name='ys3')(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7, name='ys4')(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0

# dense1 = Dense(10, name='ys1')(input1)     # 이름명시
# _____________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  ys1 (Dense)                 (None, 10)                40

#  ys2 (Dense)                 (None, 9)                 99

#  ys3 (Dense)                 (None, 8)                 80

#  ys4 (Dense)                 (None, 7)                 63

#  dense (Dense)               (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0





# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(10))
# model.add(Dense(1))

# #3.컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 에측
# loss = model.evaluate(x,y)
# result = model.predict([[10,1.3,0]])
# print('로스 : ', loss)
# print('[10,1.3,0]의 예측값 : ', result)



# (3, 10)
# (10, 3)
# 로스 :  0.004059500060975552
#[10,1.3,0]의 예측값 :  [[9.88149]]
