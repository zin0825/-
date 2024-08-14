# jena를 Dnn으로 구성

# x : (42만, 144, 144) -> (42만, 144 * 144)
# y : (42aks, 144)

# 맹그러봐.



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, Bidirectional 

import time
import numpy as np
import pandas as pd
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  

#1. 데이터
path = 'C:\\ai5\\_data\\kaggle\\jena\\'

train_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
submit = pd.read_csv(path + "jena_climate_2009_2016.csv")

# print(train_csv.shape)   # (420551, 14)

# print(train_csv.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

train_dt = pd.DatetimeIndex(train_csv.index)
# 인덱스를 데이터로 변환해서 사용할 수 있게 함
# ex) 3,4,5월은 봄인데 인덱스로 있을 경우 3,4,5월이 봄인지 알 수 없음. 겨울로 사용할 수도 있음
# 그래서 인덱스를 데이터로 변환해서 온도와 상호작용하게끔 하나 무조건은 아님. 데이터에 따라 다름

train_csv['day'] = train_dt.day
train_csv['month'] = train_dt.month
train_csv['year'] = train_dt.year
train_csv['hour'] = train_dt.hour
train_csv['dos'] = train_dt.dayofweek

# print(train_csv)   # [420551 rows x 19 columns]

y2 = train_csv.tail(144)   # tail 뒤에서 144개를 가져옴. 꼬리
y2 = y2['T (degC)']

csv = train_csv[:-144]
# csv2 = csv[:-144]
# x_pred = csv2.drop(['T (degC)', 'sh (g/kg)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'], axis=1)  
# 위의 스플릿전에서 미리 144를 정답지로 넣음 
# print(csv.shape)    # (420407, 14) <- 144개를 없앰. / (420407, 19)


x = train_csv.drop(['T (degC)', 'sh (g/kg)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'], axis=1)    
y = train_csv['T (degC)']

# print(x.shape)   # (420551, 14)
# print(y.shape)   # (420551,)


size = 144

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

x = x.reshape(420407, 144 * 14) 
x_pred = x_pred.reshape(1, 144 * 14)
# x_pred = np.array(x_pred).reshape(1, 144, 14) 저놈이 이 shpae에 맞춰서 변경

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle= True,
                                                    random_state=131)



# 2. 모델
model = Sequential()
# model.add(LSTM(32, return_sequences=True, input_shape=(144, 14)))   # timesteps, features / activation='tanh'
# model.add(LSTM(120))
# model.add(Conv1D(filters=32, kernel_size=7, input_shape=(144, 14)))
# model.add(Conv1D(120, 7))
# model.add(Flatten())   # Flatten은 2차원을 1차원으로 바꿔줌 / 이거 넣어야 돌아감

model.add(Dense(32, activation='relu', input_shape=(144 * 14,)))
model.add(Dense(220, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(370, activation='relu'))
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


model.fit(x_train, y_train, epochs=3000, batch_size=2224, 
          verbose=1, 
          validation_split=0.3, 
          callbacks=[es])


path2 = 'C:\\ai5\\_save\\keras61\\'
model.save(path2 + 'keras61_jena_05.h5')


end = time.time()


#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=510)

y_pred = model.predict(x_pred, batch_size=510) 



y_pred = y_pred.T   # reshape 대신에


print('[144]의 결과 : ', y_pred)
print('로스 : ', loss)
print('걸린 시간 : ', round(end - start, 2), "초")




def RMSE(y_test, y_pred):   # 여기있는 함수 정의라서 호출 될 때까지 실행하진 않고 호출 할 때 실행함 / 여기 pred는 무조건 세트고 위에 있는 pred랑 입력값으로 받음
    return np.sqrt(mean_squared_error(y_test, y_pred))   # np.sqrt 루트를 씌우다
rmse = RMSE(y2, y_pred)   # 내가 사용할 함수 명을 사용 / 여기서 입력을 받음, 호출 및 실행
print("RMSE : ", rmse)
# 매개 변수 (입력 받을 값)



# submit = submit[['Date Time','T (degC)']]
# submit = submit.tail(144)
# print(submit)

# # y_submit = pd.DataFrame(y_predict)
# # print(y_submit)

# submit['T (degC)'] = y_pred
# # print(submit)                  # [6493 rows x 1 columns]
# # print(submit.shape)            # (6493, 1)

# submit.to_csv(path2 + "jena_최진영_07.csv", index=False)





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


# 로스 :  [0.40843600034713745, 0.11403948813676834]
# 걸린 시간 :  1042.08 초
# RMSE :  0.7847969258434951
# jena_최진영_07.h5


# Conv1D
# 로스 :  [12.316596031188965, 0.006089337170124054]
# 걸린 시간 :  67.52 초
# RMSE :  3.591258616675911
# jena_최진영_09.h5

# epochs=1000
# 로스 :  [13.299722671508789, 0.006393804214894772]
# 걸린 시간 :  77.8 초
# RMSE :  3.7156264721195633
# jena_최진영_10.h5


# model.add(Conv1D(filters=32, kernel_size=7, input_shape=(144, 14)))
# model.add(Conv1D(120, 7))
# epochs=2000
# # 로스 :  [10.879790306091309, 0.014681011438369751]
# 걸린 시간 :  161.41 초
# RMSE :  3.3646938003108198
# jena_최진영_11.h5

# 로스 :  [4.09359884262085, 0.03151945769786835]
# 걸린 시간 :  242.26 초
# RMSE :  2.695003680187871
# jena_최진영_12.h5


# epochs=3000
# 로스 :  [5.93628454208374, 0.03351276367902756]
# 걸린 시간 :  125.55 초
# RMSE :  2.1290367396476486
# jena_최진영_13.h5


# model.add(Dense(32, activation='relu', input_shape=(144 * 14,)))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(370, activation='relu'))
# model.add(Dropout(0.1)) 
# model.add(Dense(320, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(144))


# Dnn
# 로스 :  [10.998258590698242, 0.006155939307063818]
# 걸린 시간 :  32.17 초
# RMSE :  4.266120480050729
# keras61_jena_01.h5

# 로스 :  [10.55929946899414, 0.009445608593523502]
# 걸린 시간 :  66.44 초
# RMSE :  3.3446215270025683
# keras61_jena_02.h5

# 로스 :  [8.602106094360352, 0.012383238412439823]
# 걸린 시간 :  42.89 초
# RMSE :  2.338547553325239
# keras61_jena_03.h5

# model.add(Dense(32, activation='relu', input_shape=(144 * 14,)))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(320, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(370, activation='relu'))
# model.add(Dropout(0.1)) 
# model.add(Dense(320, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(144))

# 로스 :  [8.640109062194824, 0.012897025793790817]
# 걸린 시간 :  43.43 초
# RMSE :  2.3858331536039494
# keras61_jena_04.h5

# epochs=4000 별루 -> 저장 x


# model.add(Dense(32, activation='relu', input_shape=(144 * 14,)))
# model.add(Dense(220, activation='relu'))
# model.add(Dense(320, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dense(370, activation='relu'))
# model.add(Dense(320, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(144))
# epochs=3000

# keras61_jena_05.h5


