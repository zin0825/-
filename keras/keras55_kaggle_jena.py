# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data


# y는 T (degC) 로 잡아라.

# 자르는거는 맘대로
# 온도니까 회귀모델로 하면 됨
# 십분 (마다 나눠진) 데이터
# 프레딕트로 잡아야할 곳 2016 12 31 이만큼의 데이터는 쓰면 안된다 총 144개의 데이터

# 31.12.2016 00:10:00 부터
# 01.01.2017 00:00:00 까지

# 맞춰라 !!! 

# x는 (419832, 720, 13)
# y의 shape는 (n, 144) 두개의 크기가 틀리는데 크기를 맞춰야함
# 프레딕은 (1, 144)


import numpy as np
import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import r2_score, mean_squared_error    
import os


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"   


#1. 데이터

path = 'C:\\프로그램\\ai5\\_data\\kaggle\\jena\\'

train_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
submit = pd.read_csv(path + 'jena_climate_2009_2016.csv')
print(train_csv.shape)   # (420551, 14)


print(train_csv.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')


y2 = train_csv.tail(144)   # tail 뒤에서 144개를 가져옴. 꼬리
y2 = y2['T (degC)']

csv = train_csv[:-144]


x = train_csv.drop(['T (degC)'], axis=1)

y = train_csv['T (degC)']

print(x.shape)   # (420407, 13)
print(y.shape)   # (420407,)


x_size = 144
y_size = 144

def split_x(dataset, size):   # split_x 함수를 정의하고 매개변수를 넣겠다 / split 함수를 정의 한다
    aaa = []   # aaa라는 빈 리스트
    for i in range(len(dataset) - size + 1):   # len = 길이 / dataset의 갯수 만큼 for문을 돌려서 dataset의 길이에서 size를 뺀 값에 1을 더한 만큼 반복
        subset = dataset[i : (i + size)]   # 반복 할 때 사용되는 데이터의 길이(수)는 여기서 결정
                                           # subset은 dataset의 i부터 i+size 까지 넣겠다. 1회전 할 때 i는 0이므로 0 + 5 = 5 마지막 숫자는 5

        aaa.append(subset)   # append 추가하다 / aaa에 subset을 추가하겠다 
    return np.array(aaa)   # 넘파이 배열로 변환


x = split_x(x, x_size)   # 스플릿_x라는 함수에 a와 사이즈를 입력 / split를 표출해준다
# print(x)
print(x.shape)   # (419688, 720, 13)

y = split_x(y, y_size)
# print(y)
print(y.shape)   # (420264, 144)


# x = x[:-1]
# y = y[1:]
# x_predict = x[-1:]   # 위의 스플릿에서 144를 넣었기에 1로 대체됨 (1에 144개의 데이터가 들어갔음)


x = x[:-1, :]
y = y[1:]  
  
   
# print(x, y)
print(x.shape, y.shape)   # (420263, 144, 13) (420263, 144)




x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True, 
                                                    random_state=131)


#2. 모델 구성
model = Sequential()
model.add(LSTM(30, return_sequences=True, input_shape=(144, 13)))   # shape 차원 넣어주기
model.add(LSTM(30)) 
model.add(Dense(30, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(144))


# # model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True)



# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# import datetime   # 날짜
# date = datetime.datetime.now()   # 현재 시간
# print(date)   # 2024-07-26 16:50:13.613311
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
# print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
# print(type(date))



# path = './_save/keras55_01/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# filepath = "".join([path, '/k55_01_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# ######################### cmp 세이브 파일명 만들기 끗 ###########################




# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True, # 가장 좋은 놈을 저장
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )   # 파일네임, 패스 더하면 요놈


# hist = model.fit(x_train, y_train, epochs=50, batch_size=5500,
#           verbose=1, 
#           validation_split=0.3,
#           callbacks=[es, mcp])

# print(x_train.shape, y_train.shape)


model.fit(x_train, y_train, epochs=100, batch_size=1024,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es])


np_path = './/_data//_save//keras55//'
model.save(".//_data//_save///keras55//k55_07_.h5")


end = time.time()


#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1320)


y_pred = model.predict(x, batch_size=1320)

y_pred = y_pred.T

# y_pred = np.array(y_pred).reshape(144, 1)   # 1은 가로로 있는 것을 세로로
# print(y_pred.shape)

# acc = accuracy_score(y_cor, y_pred)
# print('acc : ', acc)

print('[144]의 결과 :', y_pred)
print('로스 : ', loss)
print('걸린 시간 : ', round(end - start,2), "초")



def RMSE(y_test, y_pred):   # 여기 있는 함수 정의라서 호출 될 때까지 실행하진 않고 호출 할 때 실햄함 / pred는 무조건 세트고 위에 있는 pred랑 입력값으로 받음
      return np.sqrt(mean_squared_error(y_test, y_pred))   # np.sqrt 루트를 씌우다
rmse = RMSE(y2, y_pred)   # 내가 사용할 함수 명을 사용/ 여기서 입력을 받음, 호출 및 실행
print("RMSE : ", rmse)
# 매개 변수 (입력 받을 값)


submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = y_pred
print(submit)                  # [6493 rows x 1 columns]
print(submit.shape)            # (6493, 1)

submit.to_csv(path + "jena_최진영_k55_07.csv", index=False)



# model.add(LSTM(30, return_sequences=True, input_shape=(144, 13))) 
# model.add(LSTM(30)) 
# model.add(Dense(30, activation='relu'))
# model.add(Dense(26, activation='relu'))
# model.add(Dense(26, activation='relu'))
# model.add(Dense(22, activation='relu'))
# model.add(Dense(22, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(144))

# epochs=50
# 로스 :  9.961748123168945
# 걸린 시간 :  239.91 초
# RMSE :  2.4399428060090322
# k55_01_.h5

# 로스 :  9.725374221801758
# 걸린 시간 :  240.55 초
# RMSE :  2.8291819247740935
# k55_02_.h5


# epochs=550, batch_size=5500
# 로스 :  1.568070888519287
# 걸린 시간 :  1026.51 초
# RMSE :  1.9054759149418996
# k55_03_.h5


# epochs=1050, batch_size=5500
# 로스 :  70.85763549804688
# 걸린 시간 :  473.23 초
# RMSE :  11.404735555324553
# k55_04_.h5


# epochs=550,  batch_size=5500
# 로스 :  5.0719170570373535
# 걸린 시간 :  865.43 초
# RMSE :  2.03858837142532
# np_path = './/_data//_save//keras55//'
# model.save(".//_data//_save///keras55//k55_05_.h5")


# epochs=2550, batch_size=432
# 로스 :  0.1952672004699707
# 걸린 시간 :  3231.16 초
# RMSE :  0.3115642101079219
# k55_06_.h5

# 로스 :  0.25904014706611633
# 걸린 시간 :  3293.17 
# RMSE:  0.3839224008198416
# jena_최진영_k55_07.csv

