# 89이상
# 다중분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



#1. 데이터
path = ".\\_data\\kaggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)


# train_csv.boxplot()
# plt.show()

# print(train_csv.info())

# print(train_csv.describe())

# x와 y 분류
x = train_csv.drop(['feat_24','feat_73','feat_74'], axis=1).copy()
y = train_csv[['feat_24','feat_73','feat_74']]


################### Population x 로그 변환 ###################

train_csv[['feat_24','feat_73','feat_74']] = np.log1p(train_csv[['feat_24','feat_73','feat_74']])
# 지수변환 np.exp1m / 로그, 지수 짝이 맞아야함

##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=393)

####################### y 로그 변환 ##########################

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

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
