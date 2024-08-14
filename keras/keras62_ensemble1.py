import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate   # 소문자, 대문자 다름 주의
# from keras.layers.merge import Concatenate   # 버전에 따라 위치가 다름. 성능은 똑같음
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터           # 100 바이 2로 해주고 싶었지만 데이터를 잘못줘서 반대로 하기 위해 .T를 붙임 
x1_dataset = np.array([range(100), range(301, 401)]).T   # (2, 100) -> (100 ,2)
                    # 삼성 종가, 하이닉스 종가
x2_dataset = np.array([range(101, 201), range(411, 511),
                       range(150, 250)]).transpose()
                    # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔


x3_dataset = np.array([range(100, 105), range(401, 406)]).T   # (2, 100) -> (100 ,2)
                    # 삼성 종가, 하이닉스 종가
x4_dataset = np.array([range(201, 206), range(511, 516),
                       range(250, 255)]).transpose()
                    # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔



y = np.array(range(3001, 3101))   # 한강의 화씨 온도. 위에 (이거)는 이거야

# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_dataset, x2_dataset, y,
#                                                                          train_size=0.8, 
#                                                                          shuffle=True,
#                                                                          random_state=131)

# print(x1_train.shape, x1_test.shape)   # (80, 2) (20, 2)
# print(x2_train.shape, x2_test.shape)   # (80, 3) (20, 3)
# print(y_train.shape, y_test.shape)   # (80,) (20,)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_dataset, x2_dataset, y, 
                                                                         train_size=0.7, 
                                                                         random_state=777)

print(x1_train.shape, x2_train.shape, y_train.shape, y_test.shape)   # (70, 2) (70, 3) (70,)


#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output1 = Dense(50, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)

# model1.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 2)]               0

#  bit1 (Dense)                (None, 10)                30

#  bit2 (Dense)                (None, 20)                220

#  bit3 (Dense)                (None, 30)                630

#  bit4 (Dense)                (None, 40)                1240

#  bit5 (Dense)                (None, 50)                2050

# =================================================================
# Total params: 4,170
# Trainable params: 4,170
# Non-trainable params: 0


#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit21')(dense11)
output11 = Dense(300, activation='relu', name='bit31')(dense21)
# model2 = Model(inputs=input11, outputs=output11)


#2-3. 합체!!!
# merge1 = concatenate([output1, output11], name='mg1')   # 하나의 레이어에 두개의 모델을 컨텐트한 것
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)   

model = Model(inputs=[input1, input11], outputs=last_output)

# model.summary()

# merge1 = concatenate([output1, output11], name='mg1')
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']

#  bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']

#  bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']

#  bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']

#  bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']

#  mg1 (Concatenate)              (None, 350)          0           ['bit5[0][0]',
#                                                                   'bit31[0][0]']

#  mg2 (Dense)                    (None, 7)            2457        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 87,708
# Trainable params: 87,708
# Non-trainable params: 0


# merge1 = Concatenate(name='mg1')([output1, output11])
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']

#  bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']

#  bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']

#  bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']

#  bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']

#  mg1 (Concatenate)              (None, 350)          0           ['bit5[0][0]',
#                                                                   'bit31[0][0]']

#  mg2 (Dense)                    (None, 7)            2457        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 87,708
# Trainable params: 87,708
# Non-trainable params: 0


#3. 컴파일, 훈련
# 맹그러봐!!! 주의 점 2개 이상은 리스트

model.compile(loss='mse', optimizer='adam', metrics='mse')

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=98,
          validation_split=0.3, verbose=1, callbacks=[es])

path = './_save/keras62_01/'
model.save(path + 'keras62_09.h5')

end = time.time()


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test, verbose=1)

y_pred = model.predict([x3_dataset, x4_dataset])

y_pred = np.reshape(y_pred, (y_pred.shape[0],))   # 나타나는게 가로냐 세로냐

print('로스 : ', loss)
print('예측값[3101:3105] : ', y_pred)
print('걸린 시간 : ', round(end - start, 2), '초')




# print('예측값 : ', y_pred[:5])
# y_pred = np.reshape(y_pred, (y_pred.shape[0],))
# 로스 :  [0.01907101832330227, 0.01907101832330227]
# 예측값 :  [3027.0486 3076.0105 3090.994  3096.987  3001.8452]
# 걸린 시간 :  22.01 초
# keras62_01.h5

# 로스 :  [0.6730623245239258, 0.6730623245239258]
# 예측값 :  [3027.016  3075.9866 3090.9775 3096.9739 3002.0312]
# 걸린 시간 :  21.72 초
# keras62_02.h5

# 로스 :  [0.14821217954158783, 0.14821217954158783]
# 예측값 :  [3026.6545 3075.61   3090.5972 3096.6614 3001.677 ]
# 걸린 시간 :  22.37 초
# keras62_03.h5


# y_pred = np.reshape(y_pred, (y_pred.shape[0],))   # 안 넣은거
# print('예측값 : ', y_pred[:5])                     # 넣은거
# 로스 :  [0.10252717137336731, 0.10252717137336731]
# 예측값 :  [[3026.9224]
#  [3075.8184]
#  [3091.0315]
#  [3097.1167]
#  [3001.9668]]
# 걸린 시간 :  21.55 초
# keras62_04.h5


# y_pred = np.reshape(y_pred, (y_pred.shape[0],))   # 안 넣은거
# print('예측값 : ', y_pred)  [:5]                   # 안 넣은거
# 로스 :  [0.2830136716365814, 0.2830136716365814]
# 예측값 :  [[3027.0967]
#  [3075.6619]
#  [3090.5696]
#  [3097.4307]
#  [3001.6443]
#  [3084.3674]
#  [3058.0327]
#  [3102.433 ]
#  [3041.5703]
#  [3028.1106]
#  [3020.0002]
#  [3006.82  ]
#  [3077.9932]
#  [3049.289 ]
#  [3061.9187]
#  [3096.139 ]
#  [3080.7654]
#  [3035.2078]
#  [3079.1338]
#  [3043.46  ]
#  [3055.1182]
#  [3086.4348]
#  [3031.152 ]
#  [3005.806 ]
#  [3011.8892]
#  [3003.7585]
#  [3063.8618]
#  [3023.0415]
#  [3056.0896]
#  [3088.5024]]
# 걸린 시간 :  19.61 초
# keras62_04.h5


# y_pred = np.reshape(y_pred, (y_pred.shape[0],))   # 넣은거
# print('예측값 : ', y_pred)  [:5]                   # 안 넣은거
# 로스 :  [0.11052121222019196, 0.11052121222019196]
# 예측값 :  [3026.96   3075.9536 3090.954  3096.9553 3001.965  3084.9539 3057.9539
#  3101.8062 3040.957  3027.9597 3019.9614 3006.9639 3077.9539 3048.9556
#  3061.9536 3095.9548 3080.9539 3034.9585 3078.954  3042.9565 3054.9546
#  3086.9536 3030.959  3005.9644 3011.963  3003.9646 3063.9536 3022.961
#  3055.9543 3088.9536]
# 걸린 시간 :  21.51 초

# 다 넣은거
# 로스 :  [0.21931549906730652, 0.21931549906730652]
# 예측값 :  [3027.0137 3075.5674 3090.7363 3097.4795 3002.5918]
# 걸린 시간 :  19.61 초
# keras62_05.h5


# x3_dataset, x4_dataset
# 로스 :  [0.33831170201301575, 0.33831170201301575]
# 예측값[3101:3105] :  [3107.6667 3112.148  3116.7937 3121.4744 3126.1555]
# 걸린 시간 :  22.02 초
# keras62_07.h5


# 로스 :  [0.26088443398475647, 0.26088443398475647]
# 예측값[3101:3105] :  [3103.3855 3105.7134 3108.0417 3110.361  3112.669 ]
# 걸린 시간 :  21.62 초
# keras62_08.h5