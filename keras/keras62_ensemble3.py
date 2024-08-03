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

x3_dataset = np.array([range(100), range(301, 401),
                       range(77, 177), range(33, 133)]).T

# 거시기1, 거시기2, 거시기3, 거시기4

y1 = np.array(range(3001, 3101))   # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))   # 비트코인 가겨.

# 맹그러봐
# 컨텐티드 하고 분기하는거 안배움


x4_dataset = np.array([range(100, 106), range(400, 406)]).T   # (2, 100) -> (100 ,2)
                    # 삼성 종가, 하이닉스 종가
x5_dataset = np.array([range(200, 206), range(510, 516),
                       range(249, 255)]).T
                    # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔
x6_dataset = np.array([range(100, 106), range(400, 406),
                       range(177, 183), range(133, 139)]).T



x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
    y_train, y_test, y2_train, y2_test = train_test_split(
        x1_dataset, x2_dataset, x3_dataset, 
        y1, y2, train_size=0.7, random_state=777)

print(x1_train.shape, x2_train.shape, x3_train.shape, 
      y_train.shape, y_test.shape)   # (70, 2) (70, 3) (70, 4) (70,) (30,)



#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output1 = Dense(50, activation='relu', name='bit5')(dense4)


#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit21')(dense11)
output11 = Dense(300, activation='relu', name='bit31')(dense21)


#2-3. 모델
input111 = Input(shape=(4,))
dense111 = Dense(100, activation='relu', name='bit111')(input111)
dense211 = Dense(200, activation='relu', name='bit211')(dense111)
output111 = Dense(300, activation='relu', name='bit311')(dense211)


#2-4. 모델 병합
merge1 = Concatenate(name='mg1')([output1, output11, output111])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
last_output2 = Dense(1, name='last2')(merge3)

model = Model(inputs=[input1, input11, input111], outputs=[last_output, last_output2])


# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)


model.fit([x1_train, x2_train, x3_train], [y_train, y2_train], epochs=1000, batch_size=3,
          validation_split=0.3, verbose=1, callbacks=[es])


path = './_save/keras62_03/'
model.save(path + 'keras62_03_05.h5')

end = time.time()


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y_test, y2_test], verbose=1)

y_pred = model.predict([x4_dataset, x5_dataset, x6_dataset])

# y_pred = np.reshape(y_pred, (y_pred.shape[0],)) 

print('로스 : ', loss)
print('예측값[3101:3105] : ', y_pred[0], y_pred[1])
print('걸린 시간 : ', round(end - start, 2), '초')


"""
이 모델은 y 2개를 예측하는 것
첫번째 로스는 전체 (종합)로스,  두번째 로스는 y1에 대한 로스, y2에 대한 로스
metrics='mse' 있으면 로스가 여러개 나옴. 안넣으면 3개 나와야함
"""



# x4_dataset = np.array([range(100, 105), range(401, 406)]).T   # (2, 100) -> (100 ,2)
#                     # 삼성 종가, 하이닉스 종가
# x5_dataset = np.array([range(201, 206), range(511, 516),
#                        range(250, 255)]).transpose()
#                     # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔
# x6_dataset = np.array([range(100, 105), range(401, 406),
#                        range(177, 182), range(133, 138)]).T



# epochs=1000, batch_size=98
# 로스 :  [5026673.0, 2799955.25, 2226717.5, 2799955.25, 2226717.5]
# 예측값[3101:3105] :  [[5504.2163]
#  [5522.111 ]
#  [5540.0034]
#  [5557.898 ]
#  [5575.7905]] [[15169.17 ]
#  [15217.64 ]
#  [15266.11 ]
#  [15314.582]
#  [15363.051]]
# 걸린 시간 :  2.24 초
# keras62_03_01.h5


# metrics='mse'
# epochs=1000, batch_size=8
# 로스 :  [48.56193161010742, 2.1825482845306396, 46.3793830871582, 2.1825482845306396, 46.3793830871582]
# 예측값[3101:3105] :  [[3115.3076]
#  [3122.674 ]
#  [3130.042 ]
#  [3137.41  ]
#  [3144.7778]] [[13165.503]
#  [13195.825]
#  [13226.157]
#  [13256.491]
#  [13286.827]]
# 걸린 시간 :  7.67 초
# keras62_03_02.h5


# metrics='mse' xxx
# batch_size=3
# 로스 :  [14.260804176330566, 0.4908919036388397, 13.769912719726562]
# 예측값[3101:3105] :  [[3108.3774]
#  [3113.0635]
#  [3117.751 ]
#  [3122.344 ]
#  [3126.9377]] [[13138.193 ]
#  [13156.807 ]
#  [13175.423 ]
#  [13194.6455]
#  [13213.974 ]]
# 걸린 시간 :  11.22 초
# keras62_03_03.h5

# 로스 :  [8.03477954864502, 0.30147096514701843, 7.733308792114258]
# 예측값[3101:3105] :  [[3107.0415]
#  [3111.6914]
#  [3116.5637]
#  [3121.5432]
#  [3126.5647]] [[13131.563]
#  [13150.266]
#  [13169.832]
#  [13189.813]
#  [13209.951]]
# 걸린 시간 :  39.59 초
# keras62_03_04.h5




# x4_dataset = np.array([range(100, 106), range(400, 406)]).T   # (2, 100) -> (100 ,2)
#                     # 삼성 종가, 하이닉스 종가
# x5_dataset = np.array([range(200, 206), range(510, 516),
#                        range(249, 255)]).T
#                     # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔
# x6_dataset = np.array([range(100, 106), range(400, 406),
#                        range(177, 183), range(133, 139)]).T



# keras62_03_05.h5