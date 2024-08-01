import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\keggle\\bike-sharing-demand\\'   # 절대경로, 슬래시a, 슬래시t 특수문자로 경로 틀어짐
# path = 'C://프로그램//ai5//_data//bike-sharing-demand//'   # 절대경로
# path = 'C:/프로그램/ai5/_data/bike-sharing-demand/'   # 절대경로
# / // \ \\ 다 가능.
# 슬래시a, 슬래시b는 노란거 뜨는걸로 확인 특수문자로 인식

train_csv = pd.read_csv(path + "train.csv", index_col=0)  # pd는 판다스
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)   # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(sampleSubmission_csv.shape)   # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')   # 11개, 'casual', 'registered' 여기가 문제 두개 더하면 count임

print(train_csv.info())
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64
# dtypes: float64(3), int64(8)

print(test_csv.info())
#  0   season      6493 non-null   int64
#  1   holiday     6493 non-null   int64
#  2   workingday  6493 non-null   int64
#  3   weather     6493 non-null   int64
#  4   temp        6493 non-null   float64
#  5   atemp       6493 non-null   float64
#  6   humidity    6493 non-null   int64
#  7   windspeed   6493 non-null   float64
# dtypes: float64(3), int64(5)

print(train_csv.describe())   # 묘사 이상치가 있으면 평균의 오류로 이상치 찾아야함
#              season       holiday    workingday  ...        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  ...  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875  ...     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159  ...     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000  ...      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000  ...      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000  ...     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000  ...     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000  ...    367.000000    886.000000    977.000000
# std  표준편차 , 25 50 70% 중위값 
# shape -> info -> describe 로 데이터 확인




########### 결측치 확인 ############
print(train_csv.isna().sum())
# [8 rows x 11 columns]
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0

print(train_csv.isnull().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
 
print(test_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
print(test_csv.isnull().sum())

########## x 와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# 봤더니 캐주얼과 레지스터가 있었는데 그게 카운터였다
# [] 파이썬 리스트, 여러개의 열을 대괄호 묶어서 하나로.
# 두 개 이상은 리스트, 한개짜리는 리스트 안해도 됨
# x = train_csv.drop('casual', axis = 1) 리스트가 아닌 것
print(x)   # [10886 rows x 8 columns] 

y = train_csv['count']
print(y)
# 2011-01-01 00:00:00     16
# 2011-01-01 01:00:00     40
# 2011-01-01 02:00:00     32
# 2011-01-01 03:00:00     13
# 2011-01-01 04:00:00      1
#                       ...
# 2012-12-19 19:00:00    336
# 2012-12-19 20:00:00    241
# 2012-12-19 21:00:00    168
# 2012-12-19 22:00:00    129
# 2012-12-19 23:00:00     88

print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=96)

print(x)
print(y)
print(x.shape, y.shape)   # (10886, 8) (10886,)

#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))   # 한정시킬거야, relu 음수는 0, 양수는 그래도
model.add(Dense(80, activation='relu'))   # 레이어는 깊은데 다 relu하고 싶을땐 
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))   # 위에서 다 처리했다 생각하고 마지막은 linear 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# 로스 :  24937.365234375
# r2스코어 :  0.24071655984825868

# 로스 :  23915.240234375
# r2스코어 :  0.2728351917142603


y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   # (6493, 1)

sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
print(sampleSubmission_csv.shape)
# [6493 rows x 1 columns]
# (6493, 1)

sampleSubmission_csv.to_csv(path + "sampleSubmission_0719_0935.csv")

print('로스 : ', loss)


# 로스 :  23966.982421875
# r2스코어 :  0.27126187927641887


# train_size=0.9,
# shuffle=True,
# random_state=5454
# model.add(Dense(100, input_dim=8))
# model.add(Dense(400))
# model.add(Dense(300))
# model.add(Dense(60))
# model.add(Dense(1))
# epochs=600, batch_size=150
# 로스 :  23957.0546875
# r2스코어 :  0.2715638615615851

# 로스 :  24362.451171875
# r2스코어 :  0.2592372629050874

# 로스 :  24969.369140625
# r2스코어 :  0.24078327115766784

# train_size=0.9,
# shuffle=True,
# random_state=100
# model.add(Dense(100, activation='relu', input_dim=8)) 
# model.add(Dense(80, activation='relu'))   
# model.add(Dense(70, activation='relu'))
# model.add(Dense(35, activation='relu'))
# model.add(Dense(1, activation='linear'))
# epochs=100, batch_size=64
# 로스 :  21301.146484375
# r2스코어 :  0.31726428256680617

# model.add(Dense(90, activation='relu', input_dim=8))   
# model.add(Dense(70, activation='relu'))   
# model.add(Dense(50, activation='relu'))
# model.add(Dense(35, activation='relu'))
# model.add(Dense(1, activation='linear'))
# 로스 :  21044.263671875

# random_state=150
# model.add(Dense(200, activation='relu', input_dim=8))   # 한정시킬거야, relu 음수는 0 양수는 그래도
# model.add(Dense(80, activation='relu'))   # 레이어는 깊은데 다 relu하고 싶을땐 
# model.add(Dense(45, activation='relu'))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(1, activation='linear'))
# 로스 :  21008.8671875


# 로스 :  23950.236328125

# 로스 :  21210.515625

