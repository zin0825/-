import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



#1. 데이터
path = './_data/dacon/따릉이/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)



# train_csv.boxplot()
# plt.show()

# exit()

print(train_csv.info())
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

print(train_csv.describe())


# train_csv[['hour_bef_pm10', 'hour_bef_pm2.5', 'count']].plot.box()   # 이거 되
# plt.show()

# train_csv[['hour_bef_pm10', 'hour_bef_pm2.5', 'count']].hist(bins=50)
# plt.show()




# x와 y 분류
x = train_csv.drop(['hour_bef_pm10', 'hour_bef_pm2.5', 'count'], axis=1).copy()
y = train_csv[['hour_bef_pm10', 'hour_bef_pm2.5', 'count']]



################### Population x 로그 변환 ###################

train_csv[['hour_bef_pm10', 'hour_bef_pm2.5', 'count']] = np.log1p(train_csv[['hour_bef_pm10', 'hour_bef_pm2.5', 'count']])
# 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함

##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=57)

####################### y 로그 변환 ##########################

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

##############################################################



# 2. 모델 구성
model = RandomForestRegressor(random_state=57,
                              max_depth=5,   # 5가 디폴트
                              min_samples_split=3)   # 모두 동일한 파라미터 사용함

# model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train, )


#4. 평가 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)



# 로그 변환전 : score : 0.7592318198244835

# y 변환 후 : 0.8153940567844604

# x 변환 후 : 0.7591371600522638

# x, y 변환 후 : 0.815347846898599

# LinearRegression 변환 전 : 0.5604846037590026

# y 변환 후 : 0.6847841987152252

# x 변환 후 : 0.5608092981445729

# x, y 변환 후 : 0.6864555763521416
