# keras26_scaler05_kaggle_bike.py

# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv (카글 컴피티션 사이트)
# keras13_kaggle_bike1.py 수정

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import time

#1. 데이터
path = 'C://ai5/_data/kaggle//bike-sharing-demand/'  

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)


# train_csv.boxplot()
# plt.show()


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

print(train_csv.describe())

# train_csv[['casual', 'registered', 'count']].plot.box()   # 이거 되
# plt.show()

# train_csv[['casual', 'registered', 'count']].hist(bins=50)
# plt.show()


# x와 y 분류
x = train_csv.drop(['casual', 'registered', 'count'], axis=1).copy()
y = train_csv[['casual', 'registered', 'count']]


################### Population x 로그 변환 ###################

# train_csv[['casual', 'registered', 'count']] = np.log1p(train_csv[['casual', 'registered', 'count']])
# 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함

##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=393)

####################### y 로그 변환 ##########################

# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

##############################################################

# exit()


# # 2. 모델 구성
# model = RandomForestRegressor(random_state=393,
#                               max_depth=5,   # 5가 디폴트
#                               min_samples_split=3)   # 모두 동일한 파라미터 사용함

model = LinearRegression()



#3. 훈련
model.fit(x_train, y_train, )


#4. 평가 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)



# 로그 변환전 : score : 0.3316225983835646

# y 변환 후 : 0.3246513938752497

# x 변환 후 : 0.3316225983835646

# x, y 변환 후 : 0.3246513938752497

# LinearRegression 변환 전 : 0.2726996000531633

# y 변환 후 : 0.31913802891854154

# x 변환 후 : 0.2726996000531633

# x, y 변환 후 : 0.2726996000531633
