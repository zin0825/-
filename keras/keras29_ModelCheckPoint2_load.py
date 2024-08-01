# 이름이 카멜케이스 (자바형식)
# 모델의 어떤 지점의 체크포인트


# keras26_Scaler01_boaton
# keras29_ModelCheckPoint1


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
from matplotlib import rc



#1. 데이터 
dataset = load_boston()
x = dataset.data   # x 데이터 분리   # 스켈링 할 것, x만 (비율만) 건들고 y는 건들면 안됨
y = dataset.target   # y 데이터 분리, sklearn 문법


# x_train과 x_test 하기전에 분리...를 여기보다

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=333)

# 여기서 하는게 더 좋음. 성능이 여기서 정상적으로 잘 작동하고 위는 70%정도로 나옴

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
# 13개의 데이터를 StandardScaler 로 스켈링 한다.
scaler = MinMaxScaler()

scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)   # 위 아래 두 줄을 하나로 줄일 수 있음
x_test = scaler.transform(x_test)   # 변환된 비율만 나오는 것 




# #2. 모델 구성
# model = Sequential()
# # model.add(Dense(10, input_dim=13))   # 특성은 많으면 좋음, 한계가 있음, 인풋딤에 다차원 행렬이 들어가면 안됨 
# model.add(Dense(32, input_shape=(13,)))   # 이미지 input_shape=(8,8,1) ,하나 있는건 벡터이기 때문   # 13x10=140
# model.add(Dense(32, activation='relu'))  
# model.add(Dense(16, activation='relu'))   
# model.add(Dense(16, activation='relu'))   
# model.add(Dense(1)) 



# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=10,   # patience=참을성
#                    verbose=1,   
#                    restore_best_weights=True)

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True, # 가장 좋은 놈을 저장
#     filepath='./_save/keras29_mcp/keras29_mcp1.hdf5'   # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
#           verbose=1, 
#           validation_split=0.3,
#           callbacks=[es, mcp],   # 두개 이상은 리스트
#           )
# end = time.time()


model = load_model('./_save/keras29_mcp/keras29_mcp1.hdf5')
# 모델과 체크포인트가 저장된걸 부름



#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# print("걸린 시간 : ", round(end - start,2),'초')



# 로스 :  97.20042419433594
# r2 score :  0.008957412100013773
# 걸린 시간 :  1.84 초

# 스켈링
# 로스 :  23.58662223815918
# r2 score :  0.7595139881734306
# 걸린 시간 :  2.4 초



# 로스 :  20.869287490844727
# r2 score :  0.78721951962808
# 걸린 시간 :  5.2 초