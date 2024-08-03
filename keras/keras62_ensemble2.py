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

x4_dataset = np.array([range(100, 106), range(400, 406)]).T   # (2, 100) -> (100 ,2)
                    # 삼성 종가, 하이닉스 종가
x5_dataset = np.array([range(200, 206), range(510, 516),
                       range(249, 255)]).T
                    # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔
x6_dataset = np.array([range(100, 106), range(400, 406),
                       range(177, 183), range(133, 139)]).T

y = np.array(range(3001, 3101))   # 한강의 화씨 온도.


x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_dataset, x2_dataset, x3_dataset, y, 
                                                                         train_size=0.7, 
                                                                         random_state=777)

print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape, y_test.shape)   # (70, 2) (70, 3) (70, 4) (70,) (30,)



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
dense311 = Dense(200, activation='relu', name='bit311')(dense211)
dense411 = Dense(200, activation='relu', name='bit411')(dense311)
output111 = Dense(300, activation='relu', name='bit511')(dense411)


#2-4. 모델 병합
merge1 = Concatenate(name='mg1')([output1, output11, output111])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11, input111], outputs=last_output)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train, epochs=1000, batch_size=5,
          validation_split=0.3, verbose=1, callbacks=[es])


path = './_save/keras62_02/'
model.save(path + 'keras62_02_05.h5')

end = time.time()


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test, verbose=1)



# x4_dataset = np.array([range(100, 106), range(400, 406)]).T   # (2, 100) -> (100 ,2)
#                     # 삼성 종가, 하이닉스 종가
# x5_dataset = np.array([range(200, 206), range(510, 516),
#                        range(249, 255)]).T
#                     # 원유, 환율, 금시세  레이어 20개를 넣어서 5개로 나눌거야 그러기 위해 5개로 나눔
# x6_dataset = np.array([range(100, 106), range(400, 406),
#                        range(177, 183), range(133, 139)]).T



y_pred = model.predict([x4_dataset, x5_dataset, x6_dataset])

# y_pred = np.reshape(y_pred, (y_pred.shape[0],))

print('로스 : ', loss)
print('예측값[3101:3105] : ', y_pred)
print('걸린 시간 : ', round(end - start, 2), '초')



# batch_size=98
# 로스 :  [112061.96875, 112061.96875]
# 예측값[3101:3105] :  [3344.2197 3354.539  3364.8706 3375.203  3385.5354]
# 걸린 시간 :  1.46 초
# keras62_02_01.h5


# batch_size=8
# 로스 :  [16.500267028808594, 16.500267028808594]
# 예측값[3101:3105] :  [[3104.9443]
#  [3109.9026]
#  [3114.8616]
#  [3119.824 ]
#  [3124.776 ]
#  [3129.807 ]]
# 걸린 시간 :  3.96 초
# keras62_02_02.h5

# 로스 :  [0.8943962454795837, 0.8943962454795837]
# 예측값[3101:3105] :  [[3103.679 ]
#  [3109.1257]
#  [3114.573 ]
#  [3120.0203]
#  [3125.4675]
#  [3130.9143]]
# 걸린 시간 :  7.1 초
# keras62_02_03.h5

# batch_size=5
# 로스 :  [1.1096599102020264, 1.1096599102020264]
# 예측값[3101:3105] :  [[3100.1804]
#  [3104.6738]
#  [3109.1794]
#  [3113.716 ]
#  [3118.269 ]
#  [3122.8213]]
# 걸린 시간 :  4.65 초
# keras62_02_04.h5

# 로스 :  [0.7571924328804016, 0.7571924328804016]
# 예측값[3101:3105] :  [[3101.9631]
#  [3106.8542]
#  [3111.7703]
#  [3116.7117]
#  [3121.6562]
#  [3126.6147]]
# 걸린 시간 :  11.29 초
# keras62_02_05.h5