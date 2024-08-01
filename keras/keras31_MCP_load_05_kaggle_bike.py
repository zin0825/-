import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import rc

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
# 11개, 'casual', 'registered' 여기가 문제 두개 더하면 count임

print(train_csv.info())

print(test_csv.info())

print(train_csv.describe())   # 묘사


########### 결측치 확인 ############
print(train_csv.isna().sum())

print(train_csv.isnull().sum())

print(test_csv.isna().sum())

print(test_csv.isnull().sum())

########## x 와 y를 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
# 봤더니 캐주얼과 레지스터가 있엇는데 그게 카운터였다
# [] 파이썬 리스트, 대괄호 묶어서 하나하나하나, 앞에는 인식하지만 뒤에는 인식 안됨
# 두 개 이상은 리스트, 한개짜리는 리스트 안해도 됨
print(x)   # [10886 rows x 8 columns] 

y = train_csv['count']
print(y)

print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=8150)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

test_csv = scaler.transform(test_csv)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_05_kaggle_bike_save.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_05_kaggle_bike_save.hdf5')  

loss2 = model2.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)



############################## 원래값 ###########################
# 로스 :  21045.46484375
# 걸리시간 :  6.66 초

# ======================== 1. save_model 출력 ====================
# 로스 :  21045.46484375
# r2 score :  0.30498467625064996
# ======================== 2. MCP 출력 ====================
# 로스 :  21045.46484375
# r2 score :  0.30498467625064996