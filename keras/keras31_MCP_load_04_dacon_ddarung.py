import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt

#Hello 준영 I

#1. 데이터
path = './_data/dacon/따릉이/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

print(train_csv.shape)   # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission_csv.shape)   # (715, 1)

print(train_csv.columns)

print(train_csv.info())
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)   # [1328 rows x 10 columns]

print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)   # [1328 rows x 9 columns]

y = train_csv['count']
print(y)
print(y.shape)   # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_04_ddarung_save.hdf5')  


loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_04_ddarung_save.hdf5')
                    
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)


############################## 원래값 ###########################
# 로스 :  1581.5755615234375
# 걸린시간 :  5.97 초
# model = load_model('./_save/keras30_mcp/keras30_04_dacon_ddarung_save.hdf5')  

# ======================== 1. save_model 출력 ====================
# 로스 :  1581.5755615234375
# r2 score :  0.7501257537925873
# ======================== 2. MCP 출력 ====================
# 로스 :  1581.5755615234375
# r2스코어 :  0.7501257537925873