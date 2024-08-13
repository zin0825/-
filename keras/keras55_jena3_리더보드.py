# 리더보드 테스트용

학생csv = 'jena_최진영.csv'

path1 = 'C:\\ai5\\_data\\_kaggel\\jena\\'   # 원본csv 데이터 저장위치
path2 = 'C:\\ai5\\_save\\keras55'           # 가중치 파일과 생성된 csv 파일 저장위치


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  


datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)

print(datasets)
print(datasets.shape)   # (420551, 14)


# 8/12 까지 loc, iloc 알아오기


y_정답 = datasets.iloc[-144:, 1]   # 인트로케이션 원래는 인덱스로테이션
print(y_정답)
print(y_정답.shape)   # (144,)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)


print(y_정답[:5])
print(학생꺼[:5])


def split_x(dataset, size):   # split_x 함수를 정의하고 매개변수를 넣겠다 / split 함수를 정의 한다
    aaa = []   # aaa라는 빈 리스트
    for i in range(len(dataset) - size + 1):   # len = 길이 / dataset의 갯수 만큼 for문을 돌려서 dataset의 길이에서 size를 뺀 값에 1을 더한 만큼 반복  
        subset = dataset[i : (i + size)]   # 반복 할 때 사용되는 데이터의 길이(수)는 여기서 결정
                                           # subset은 dataset의 i부터 i + size 까지 넣겠다. 1회전 할 때 i는 0이므로 0 + 5 = 5 마지막 숫자는 5
        aaa.append(subset)   # append 추가하다 / aaa에 subset을 추가하겠다
    return np.array(aaa)   # 넘파이 배열로 변환



x = split_x(x, size)  # 스플릿_x라는 함수에 a와 사이즈를 입렵 / split를 표출해준다
# print(x.shape)   # (420264, 144, 14)

y = split_x(y, size)
# print(y.shape)   # (420264, 144)

x_pred = x[-1]
print(x_pred.shape)   # (144, 14)
x_pred = np.array(x_pred).reshape(1, 144, 14)



x = x[:-1, :]
y = y[1:]
# 1에 144개의 데이터가 들어갔음

# print(x.shape)  # (420407, 144, 14)
# print(y.shape)  # (420407, 144)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle= True,
                                                    random_state=131)



# 2. 모델
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(144, 14)))   # timesteps, features / activation='tanh'
model.add(LSTM(120))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(370, activation='relu'))
model.add(Dropout(0.1)) 
model.add(Dense(320, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(288, activation='relu'))
model.add(Dense(144))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True)



# # #################### mcp 세이프 파일명 만들기 끗 #####################
# # import datetime
# # date = datetime.datetime.now()
# # # print(date) # 2024-07-26 16:51:36.578483
# # # print(type(date))
# # date = date.strftime("%m%d_%H%M")
# # # print(date) # 0726 / 0726_1654
# # # print(type(date))

# # path = './_save/keras55/'
# # filename = '{epoch:04d}-{loss:4f}.hdf5' # '1000-0.7777.hdf5'
# # filepath = "".join([path, 'jena_최진영_01_', date, '_', filename])
# # # 생성 예: "./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"

# # #################### mcp 세이프 파일명 만들기 끗 #####################

# # mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
# #     monitor='val_loss',
# #     mode='auto',
# #     verbose=1,
# #     save_best_olny=True,
# #     filepath = filepath,
# # )


model.fit(x_train, y_train, epochs=1000, batch_size=2224, 
          verbose=1, 
          validation_split=0.3, 
          callbacks=[es])


path2 = 'C:\\ai5\\_save\\keras55\\'
model.save(path2 + 'jena_최진영_03.h5')


end = time.time()


#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=510)

y_pred = model.predict(x_pred, batch_size=510) 



y_pred = y_pred.T   # reshape 대신에


print('[144]의 결과 : ', y_pred)
print('로스 : ', loss)
# print('걸린 시간 : ', round(end - start, 2), "초")




def RMSE(y_test, y_pred):   # 여기있는 함수 정의라서 호출 될 때까지 실행하진 않고 호출 할 때 실행함 / 여기 pred는 무조건 세트고 위에 있는 pred랑 입력값으로 받음
    return np.sqrt(mean_squared_error(y_test, y_pred))   # np.sqrt 루트를 씌우다
rmse = RMSE(y2, y_pred)   # 내가 사용할 함수 명을 사용 / 여기서 입력을 받음, 호출 및 실행
print("RMSE : ", rmse)
# 매개 변수 (입력 받을 값)



submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = y_pred
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

# submit.to_csv(path2 + "jena_최진영_03.csv", index=False)





# 로스 :  [0.39678335189819336, 0.10460339486598969]
# 걸린 시간 :  678.77 초
# RMSE :  0.841614223206231
# jena_최진영_06_.h5


# keras55
# 로스 :  [0.5734065771102905, 0.08429450541734695]
# 걸린 시간 :  754.04 초
# RMSE :  0.7112192808254708
# jena_최진영_01_01.h5

# 로스 :  [100.17037963867188, 0.00588001636788249]
# 걸린 시간 :  11.89 초
# RMSE :  17.665175706773116





# jena_최진영_01.csv