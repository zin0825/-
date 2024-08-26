import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd



#1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)


# df.boxplot()
# plt.show()


# exit()

print(df.info())
#  0   age     442 non-null    float64
#  1   sex     442 non-null    float64
#  2   bmi     442 non-null    float64
#  3   bp      442 non-null    float64
#  4   s1      442 non-null    float64
#  5   s2      442 non-null    float64
#  6   s3      442 non-null    float64
#  7   s4      442 non-null    float64
#  8   s5      442 non-null    float64
#  9   s6      442 non-null    float64
#  10  target  442 non-null    float64

print(df.describe())


# df['target'].hist(bins=50)
# plt.show()


# x와 y 분류
x = df.drop(['target'], axis=1).copy()
y = df['target']



################### Population x 로그 변환 ###################



##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)

######################## y 로그 변환 ##########################

# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

###############################################################

# exit()


#2. 모델 구성
# model = RandomForestRegressor(random_state=9,
#                               max_depth=5,   # 5가 디폴트
#                               min_samples_split=3)   # 모두 동일한 파라미터 사용함

model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train, )


#4. 평가 예측
score = model.score(x_test, y_test)   # r2_score와 같음

print('score : ', score)




# 로그 변환전 : score : 0.5441797366005725

# y 변환 후 : 0.4905494431647127


# LinearRegression : 0.5851141269959739

# y 변환 후 : 0.5390657026718155
