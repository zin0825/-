import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate    # 소문자, 대문자 다름 주의
# from keras.layers.merge import Concatenate   # 버전에 따라 위치가 다름. 성능은 똑같음
from keras.layers import SimpleRNN, LSTM, GRU, Conv1D, Flatten

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from IPython.display import display



#1. 데이터
path = 'C:\\ai5\\_data\\중간고사데이터\\'

voic_csv = pd.read_csv(path + '성우하이텍 240816.csv', index_col=0, encoding='cp949', thousands=',')   # thousands , 안나오게끔
naver_csv = pd.read_csv(path + 'NAVER 240816.csv', index_col=0, encoding='cp949', thousands=',')
hybe_csv = pd.read_csv(path + '하이브 240816.csv', index_col=0, encoding='cp949', thousands=',')   

# print(voic_csv.shape)   # (7058, 16)
# print(naver_csv.shape)   # (5390, 16)
# print(hybe_csv.shape)   # (948, 16)

voic_csv.info()

# exit()

print(voic_csv.isna().sum())

# # print(vois_csv.columns)
# # Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
# #        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
# #       dtype='object')


# train_dt = pd.DatetimeIndex(voic_csv.index)

# voic_csv['day'] = train_dt.day
# voic_csv['month'] = train_dt.month
# voic_csv['year'] = train_dt.year
# voic_csv['hour'] = train_dt.hour
# voic_csv['dos'] = train_dt.dayofweek



voic_csv = voic_csv[:400]
naver_csv = naver_csv[:400]
hybe_csv = hybe_csv[:400]

# print(voic_csv)   # [400 rows x 21 columns]
# exit()



x_naver = naver_csv.drop(['종가', '전일비', 'Unnamed: 6', '거래량', '금액(백만)', '개인', '기관', '외인(수량)'], axis=1)
print(x_naver.shape)   # (400, 9)

x_hybe = hybe_csv.drop(['종가', '전일비', 'Unnamed: 6', '거래량', '금액(백만)', '개인', '기관', '외인(수량)'], axis=1)
print(x_hybe.shape)   # (400, 9)

y_voic = voic_csv['종가']


print(y_voic.shape)   # (400,)

# x_naver.info()
# x_hybe.info()


# exit()

size = 20

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

print(x_naver.shape, x_hybe.shape)
# exit()
 
x_na = split_x(x_naver, size)
x_hy = split_x(x_hybe, size)
print(x_na.shape, x_hy.shape)   #  (381, 20, 8) (381, 20, 8)
# exit()

# x_na_pred = x_na[::-1]   # -1은 역순으로 바꾸는 것
x_na_pred = x_na[:1]
x_hy_pred = x_hy[:1]
print(x_na_pred.shape)   # (1, 20, 8)
print(x_hy_pred.shape)   # (1, 20, 8)


x_na = x_na[1:]   # 최신기준이 위에 있기 때문에 제일 위에 칸을 프레딕트로 버림
x_hy = x_hy[1:]
y_vo = y_voic[:-20]
print(x_na.shape)   # (380, 20, 8)
print(x_hy.shape)   # (380, 20, 8)
 


x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_na, x_hy, y_vo,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=66)



#2-1. 모델 구성
input1 = Input(shape=(20, 8))
dense1 = LSTM(260, activation='relu', name='bit1')(input1)
dense2 = Dense(260, activation='relu', name='bit2')(dense1)
dense3 = Dense(280, activation='relu', name='bit3')(dense2)
dense4 = Dense(240, activation='relu', name='bit4')(dense3)
dense5 = Dense(220, activation='relu', name='bit5')(dense4)
output1 = Dense(220, activation='relu', name='bit6')(dense5)



#2-2. 모델 2
input11 = Input(shape=(20, 8))
dense11 = LSTM(160, activation='relu', name='bit11')(input11)
dense21 = Dense(160, activation='relu', name='bit21')(dense11)
dense31 = Dense(180, activation='relu', name='bit31')(dense21)
dense41 = Dense(140, activation='relu', name='bit41')(dense31)
dense51 = Dense(120, activation='relu', name='bit51')(dense41)
output2 = Dense(120, activation='relu', name='bit61')(dense51)


#2-3. 합체
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)   

model = Model(inputs=[input1, input11], outputs=last_output)



# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit([x_na, x_hy], y_vo, epochs=3000, batch_size=8,
          validation_split=0.2, verbose=1,
          callbacks=[es])


path = 'C:\\ai5\\_save\\중간고사가중치\\'
# model.save(path + 'keras63_최진영_11.h5')
model = load_model(path + 'keras63_02_성우하이텍_최진영.h5')


# end = time.time()




#4. 평가, 예측
loss = model.evaluate([x_na, x_hy], y_vo, verbose=1)

results = model.predict([x_na_pred, x_hy_pred])

print('로스 : ', loss)
print('성우하이텍 8월19일 종가 : ', results)
# print('걸린 시간 : ', (end - start, 2), '초')




# 로스 :  1434466.0
# 예측값 :  [[7332.736]]
# 걸린 시간 :  (31.245984315872192, 2) 초
# keras63_최진영_08.h5


# 로스 :  1434466.0
# 성우하이텍 8월19일 종가 :  [[7332.736]]
