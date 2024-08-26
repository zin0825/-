from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337)   # 값 고정
np.random.seed(337)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


#1. 데이터
datasets = load_boston()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)


# df.boxplot()
# plt.show()

# exit()

print(df.info())
#  0   CRIM     506 non-null    float64
#  1   ZN       506 non-null    float64
#  2   INDUS    506 non-null    float64
#  3   CHAS     506 non-null    float64
#  4   NOX      506 non-null    float64
#  5   RM       506 non-null    float64
#  6   AGE      506 non-null    float64
#  7   DIS      506 non-null    float64
#  8   RAD      506 non-null    float64
#  9   TAX      506 non-null    float64
#  10  PTRATIO  506 non-null    float64
#  11  B        506 non-null    float64
#  12  LSTAT    506 non-null    float64
#  13  target   506 non-null    float64

print(df.describe())
#              CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT      target
# count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000
# mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.284634   68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   12.653063   22.532806
# std      8.601545   23.322453    6.860353    0.253994    0.115878    0.702617   28.148861    2.105710    8.707259  168.537116    2.164946   91.294864    7.141062    9.197104
# min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000    2.900000    1.129600    1.000000  187.000000   12.600000    0.320000    1.730000    5.000000
# 25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   45.025000    2.100175    4.000000  279.000000   17.400000  375.377500    6.950000   17.025000
# 50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500   77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   11.360000   21.200000
# 75%      3.677083   12.500000   18.100000    0.000000    0.624000    6.623500   94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   16.955000   25.000000
# max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000  100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   37.970000   50.000000


# df['B'].boxplot()   # 'Series' object has no attribute 'boxplot' 데이터프라임은 시리즈
# df['B'].plot.box()   # 이거 되
# plt.show()

# df['B'].hist(bins=50)
# plt.show()


# df['target'].hist(bins=50)
# plt.show()


# x와 y 분류
x = df.drop(['target'], axis=1).copy()
y = df['target']


################### Population x 로그 변환 ###################

x['B'] = np.log1p(x['B'])   # 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함

##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=337)

######################## y 로그 변환 ##########################

# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

###############################################################

# exit()


#2. 모델 구성
model = RandomForestRegressor(random_state=337,
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

