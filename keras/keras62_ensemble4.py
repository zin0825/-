import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate   # 소문자, 대문자 다름 주의
# from keras.layers.merge import Concatenate   # 버전에 따라 위치가 다름. 성능은 똑같음
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터           # 100 바이 2로 해주고 싶었지만 데이터를 잘못줘서 반대로 하기 위해 .T를 붙임 
x_dataset = np.array([range(100), range(301, 401)]).T   # (2, 100) -> (100 ,2)
                    # 삼성 종가, 하이닉스 종가


y1 = np.array(range(3001, 3101))   # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))   # 비트코인 가겨.

# 맹그러봐!!!


x2_dataset = np.array([range(100, 106), range(400, 406)]).T


x_train, x_test, y_train, y_test, y2_train, y2_test = train_test_split(x_dataset, y1, y2,
                                                    train_size=0.7, 
                                                    random_state=777)




#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output = Dense(50, activation='relu', name='bit5')(dense4)

# #2-4. 모델 병합
# merge1 = Concatenate(name='mg1')(output1)
# last_output = Dense(1, name='last')(merge1)
# last_output2 = Dense(1, name='last2')(merge1)

# model = Model(inputs=input1, outputs=[last_output, last_output2])


# #2-5. 분기1.
# dense51 = Dense(100, activation='relu', name='bit51')(output)   # 미들 아웃풋이 인풋
# dense52 = Dense(200, activation='relu', name='bit52')(dense51)
# dense53 = Dense(200, activation='relu', name='bit53')(dense52)
output_1 = Dense(1, activation='relu', name='output_1')(output)


# #2-6. 분기2.
# dense61 = Dense(100, activation='relu', name='bit61')(output1)   # 여기도 미들 아웃풋이 인풋
# dense62 = Dense(200, activation='relu', name='bit62')(dense61)
output_2 = Dense(1, activation='relu', name='output_2')(output)


model = Model(inputs=input1,
              outputs=[output_1, output_2])


# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   #  metrics='mse' 있으면 print(loss) 로스가 여러개 나옴.

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, [y_train, y2_train], epochs=1000, batch_size=3,
          validation_split=0.3, verbose=1)


path = './_save/keras62_04/'
model.save(path + 'keras62_04_06.h5')

end = time.time()


#4. 평가, 예측
loss = model.evaluate([x_test], [y_test, y2_test], verbose=1)

y_pred = model.predict([x2_dataset])

# y_pred = np.reshape(y_pred, (y_pred.shape[0],))

print('로스 : ', loss)
print('예측값 : ', y_pred[0], y_pred[1])
print('걸린 시간 : ', round(end - start, 2), '초')


"""
이 모델은 y 2개를 예측하는 것
첫번째 로스는 전체 (종합)로스,  두번째 로스는 y1에 대한 로스, y2에 대한 로스
metrics='mse' 있으면 로스가 여러개 나옴. 안넣으면 3개 나와야함
"""

# 로스 :  [112061.96875, 112061.96875]
# 예측값[3101:3105] :  [3344.2197 3354.539  3364.8706 3375.203  3385.5354]
# 걸린 시간 :  1.46 초
# keras62_02_01.h5

# 로스 :  [1228.0770263671875, 652.3941040039062, 575.6829833984375, 652.3941040039062, 575.6829833984375]
# 예측값[3101:3105] :  [array([[3082.281 ],
#        [3087.8044],
#        [3093.3284],
#        [3098.8518],
#        [3104.391 ]], dtype=float32), array([[13233.098],
#        [13257.203],
#        [13281.308],
#        [13305.411],
#        [13329.597]], dtype=float32)]
# 걸린 시간 :  6.26 초


# 로스 :  [2595249.0, 454862.09375, 2140387.0, 454862.09375, 2140387.0]
# 예측값[3101:3105] :  [[4247.6836]
#  [4260.9297]
#  [4274.176 ]
#  [4287.421 ]
#  [4300.667 ]] [[15340.288]
#  [15388.539]
#  [15436.793]
#  [15485.046]
#  [15533.3  ]]
# 걸린 시간 :  2.42 초
# keras62_02_02.h5


# 로스 :  [4101095.5, 1735395.25, 2365700.25, 1735395.25, 2365700.25]
# 예측값[3101:3105] :  [[5072.8906]
#  [5089.4736]
#  [5106.0645]
#  [5122.654 ]
#  [5139.245 ]] [[15666.868]
#  [15717.744]
#  [15768.662]
#  [15819.581]
#  [15870.499]]
# 걸린 시간 :  1.84 초
# keras62_02_03.h5


# epochs=1000, batch_size=3
# 로스 :  [12.1309232711792, 0.5614671111106873, 11.569456100463867]
# 예측값 :  [[3091.558 ]
#  [3092.5532]
#  [3093.5483]
#  [3094.544 ]
#  [3095.539 ]
#  [3096.5342]] [[13061.497]
#  [13062.5  ]
#  [13063.505]
#  [13064.509]
#  [13065.514]
#  [13066.517]]
# 걸린 시간 :  84.58 초
# keras62_04_04.h5

# 로스 :  [0.4804426431655884, 0.2913118302822113, 0.18913081288337708]
# 예측값 :  [[3090.7747]
#  [3091.7751]
#  [3092.7754]
#  [3093.7756]
#  [3094.776 ]
#  [3095.7761]] [[13058.666]
#  [13059.677]
#  [13060.688]
#  [13061.698]
#  [13062.708]
#  [13063.719]]
# 걸린 시간 :  85.49 초
# keras62_04_05.h5

# 로스 :  [0.08833615481853485, 0.001932630897499621, 0.08640352636575699]
# 예측값 :  [[3090.486 ]
#  [3091.148 ]
#  [3091.8096]
#  [3092.4712]
#  [3093.133 ]
#  [3093.7944]] [[13058.574]
#  [13059.668]
#  [13060.76 ]
#  [13061.852]
#  [13062.945]
#  [13064.036]]
# 걸린 시간 :  84.26 초
# keras62_04_06.h5