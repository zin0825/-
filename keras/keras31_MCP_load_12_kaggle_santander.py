# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
path = "C:\\프로그램\\ai5\\_data\\keggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)   # (200000, 201)
# print(test_csv.shape)   # (200000, 200)
# print(sample_csv.shape)   # (200000, 1)

# print(train_csv.columns)

x = train_csv.drop(['target'], axis=1)
# print(x)   # [200000 rows x 200 columns]

y = train_csv['target']
# print(y)
# print(y.shape)   # (200000,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038,
                                                    stratify=y)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (200000, 200) (200000,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

test_csv = scaler.transform(test_csv)

print("======================== 1. save_model 출력 ====================")

model = load_model('./_save/keras30_mcp/keras30_12_kaggle_santander.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=0)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


print("======================== 2. MCP 출력 ====================")

model2 = load_model('./_save/keras30_mcp/keras30_12_kaggle_santander_save.hdf5')  


loss2 = model2.evaluate(x_test, y_test, verbose=0)    
print('로스 : ', loss)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2 score : ', r2)

# acc score :  0.87495
# 걸린 시간 :  104.54 초
# 로스 :  [1.0752360820770264, 0.8749499917030334]

# batch_size=1000
# acc score :  0.897375
# 걸린 시간 :  117.65 초
# 로스 :  [0.3148635923862457, 0.8973749876022339]

# batch_size=5000
# acc score :  0.9049
# 걸린 시간 :  58.69 초
# 로스 :  [0.25515782833099365, 0.9049000144004822]

# acc score :  0.91195
# 걸린 시간 :  60.1 초
# 로스 :  [0.24282044172286987, 0.9119499921798706]

# acc score :  0.91185
# 걸린 시간 :  58.62 초
# 로스 :  [0.24183888733386993, 0.9118499755859375]

# 스켈링
# acc score :  0.912525
# 걸린 시간 :  120.47 초
# 로스 :  [0.2372458279132843, 0.9125249981880188]


# acc score :  0.913775
# 걸린 시간 :  122.54 초
# 로스 :  [0.23384465277194977, 0.9137750267982483]


############################## 원래값 ###########################
# acc score :  0.914975
# 걸린 시간 :  34.86 초
# 로스 :  [0.2327820509672165, 0.9149749875068665]

#  ======================== 1. save_model 출력 ====================
# 로스 :  [0.2327820509672165, 0.9149749875068665]
# r2 score :  0.26317500343581546
# ======================== 2. MCP 출력 ====================
# 로스 :  [0.2327820509672165, 0.9149749875068665]
# r2 score :  0.26317500343581546