# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd   # csv로 행렬 사용 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score   # r2 보조지표로 사용

#1. 데이터
path = "./_data/dacon/따릉이/"  # path = 경로 / 같은 경로를 한번에 , 상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)   
# 함수 csv파일을 불러들이겠다 / .은 루트 / ""은 문자 "1"+"A" = "1A"그냥 붙여버림
# index_col=0 = 0번째 열을 index로 표시
print(train_csv) # [1459 rows X  11colums] / id열 제외하면 10columms

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)  # [715 rows x 1 columns]   /  NaN 결측치, 데이터가 없다

print(train_csv.shape)   # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission_csv.shape)   # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',   # precipitation 비가 오는지 안오는지
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')   # 10개의 컬럼 중에 타겟은 카우트
print(train_csv.info())
#  0   hour                    1459 non-null   int64    # 진짜 데이터 1459 / 데이터의 개수가 다르면 결측치가 존재
#  1   hour_bef_temperature    1457 non-null   float64   
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64   non-null - NaN 같은놈 
#  6   hour_bef_ozone          1383 non-null   float64   non-null - y에 결측치가 있으면 안됨 훈련이 안됨
#  7   hour_bef_pm10           1369 non-null   float64   데이터가 많을 땐 수정보단 삭제가 더 나음
#  8   hour_bef_pm2.5          1342 non-null   float64   
#  9   count                   1459 non-null   float64   = 데이터 10개
# dtypes: float64(9), int64(1)

#################### 결측치 처리 1. 삭제 #########################
# print(train_csv.isnull().sum())   # 결측치가 있냐
print(train_csv.isna().sum())   # isnull() 위와 동일
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0

train_csv = train_csv.dropna()   # 데이터를 떨군다, 삭제
# 결측치가 있는 컬럼만 삭제하는게 아님, 컬럼의 크기는 동일해야한다, dropna() 얘를 기준으로 쭉쭉 지워짐
print(train_csv.isna().sum())   # 결측치가 있니, 결측치가 없어요, 결측치를 전부 더해줘(+)
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0

print(train_csv)   # [1328 rows x 10 columns]   # 지워서 1328
print(train_csv.isna().sum())   # isna() 결측치를 확인하고, sum() 결측치의 개수를 확인
print(train_csv.info())
#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64


print(test_csv.info())   # 훈련 안하고 평가만 할 것, 결측치를 삭제할 수 없음 서브미션에 넣어야하기 때문, 결측치에 평균값 넣자
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64   # NaN 수치 데이터가 아님
#  2   hour_bef_precipitation  714 non-null    float64   # 지금 하는게 #1,2,3,4 중에 #1 데이터,정제
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64

test_csv = test_csv.fillna(test_csv.mean())   # fillna 결측치를 채워넣기, 컬럼별 mean 평균을 집어넣는다
print(test_csv.info())
#  0   hour                    715 non-null    int64   # 이빨빠진 놈들은 평균으로 매꿨다 (컬럼들끼리의 평균)
#  1   hour_bef_temperature    715 non-null    float64
#  2   hour_bef_precipitation  715 non-null    float64
#  3   hour_bef_windspeed      715 non-null    float64
#  4   hour_bef_humidity       715 non-null    float64
#  5   hour_bef_visibility     715 non-null    float64
#  6   hour_bef_ozone          715 non-null    float64
#  7   hour_bef_pm10           715 non-null    float64
#  8   hour_bef_pm2.5          715 non-null    float64


x = train_csv.drop(['count'], axis=1)
# .drop 컬럼 하나를 삭제 /axis= 축(중심선) = 행, 행과 열 따라 동작
# 카운터(행) 를 지운다 xx -> x를 지우는게 아니라 카운트만 빼고 카피해서 넣는다

print(x)   # [1328 rows x 9 columns]

y = train_csv['count']   # train_csv만 y에 넣어줘, train_csv['열이름']으로 해당 열의 데이터를 가져옴
"""
print(y)
# id
# 3        49.0
# 6       159.0
# 7        26.0
# 8        57.0
# 9       431.0
#         ...
# 2174     21.0
# 2175     20.0
# 2176     22.0
# 2178    216.0
# 2179    170.0
# Name: count, Length: 1328, dtype: float64
"""
print(y.shape)   # (1328,)  / 여기까지가 전처리 부분

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)
# 트레인에서 훈련한 걸 다시 분리한다

print(x.shape, y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(600, activation='relu', input_dim=9))
model.add(Dense(550, activation='relu'))
model.add(Dense(139, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32)  #batch_size는 열

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)   # 여기는 훈련이 아님

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# random_state=250)
# model.add(Dense(1, input_dim=9))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))
# batch_size=10

# 로스 :  2747.318359375
# r2스코어 :  0.5756438105775017   #r2와 로스 모두 올라갈 경우 r2는 보조이기 때문에 크게 상관 xx 

# 로스 :  2687.07177734375
# r2스코어 :  0.5849496663355308

# 로스 :  2719.63232421875
# r2스코어 :  0.5799202642210031

# train_size=0.9,   # train_size=0.7 이걸 바꿔야 로스값이 잘 줄어듦
# shuffle=True,  
# random_state=4343
# model.add(Dense(55, input_dim=9))
# model.add(Dense(77))
# model.add(Dense(66))
# model.add(Dense(5))
# model.add(Dense(1))
# model.fit(x_train, y_train, epochs=300, batch_size=10)   # 데이터가 클수록 배치를 크게 주는게 좋음
# 로스 :  1730.202880859375     # 1번
# r2스코어 :  0.688370212310059


y_submit = model.predict(test_csv)   # submission에 test_csv 예측값 넣기
print(y_submit)
print(y_submit.shape)  # (715, 1)


##### submission.csv 만들기 // count컬럼에 값 만 너주면 되 #####
submission_csv['count'] = y_submit   # y_submit를 ['count']에 집어넣음
print(submission_csv)
print(submission_csv.shape)   
# [715 rows x 1 columns]
# (715, 1)
print('로스 : ', loss)

submission_csv.to_csv(path + "submission_0717_2021.csv")   # 넣기, 이대로 하면 그래도 덮어씌어짐





# model = Sequential()
# model.add(Dense(100, input_dim=9))
# model.add(Dense(88))
# model.add(Dense(55))
# model.add(Dense(5))
# model.add(Dense(1))
# 로스 :  1694.266357421875    # 2번
# r2스코어 :  0.6948428246468762


# model.add(Dense(200, input_dim=9))
# model.add(Dense(366))
# model.add(Dense(256))
# model.add(Dense(110))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))
# pochs=300, batch_size=32
# 로스 :  1692.836181640625    # 3번
# r2스코어 :  0.6951003835058333



# train_size=0.98,
# shuffle=True,
# random_state=5757
# model.add(Dense(500, input_dim=9))
# model.add(Dense(250))
# model.add(Dense(125))
# model.add(Dense(72.5))
# model.add(Dense(50))
# model.add(Dense(1))
# epochs=1000, batch_size=32
# 로스 :  1074.052490234375    # 4번
# r2스코어 :  0.8203732823180739

# train_size=0.98,
# shuffle=True,
# random_state=575
# model.add(Dense(500, input_dim=9))
# model.add(Dense(250))
# model.add(Dense(125))
# model.add(Dense(79))
# model.add(Dense(50))
# model.add(Dense(1))
# epochs=1000, batch_size=32
# 로스 :  965.5760498046875
# r2스코어 :  0.8385150831694692

# 로스 :  1119.56787109375


# 

