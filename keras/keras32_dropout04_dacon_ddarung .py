import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

print(x)
print(y)
print(x.shape, y.shape)   # (1328, 9) (1328,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.007490636704119841 1.0




#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=9))
model.add(Dropout(0.3)) 
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(79, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()


from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss',
                   mode= 'min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)

######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   # 현재 시간
print(date)   # 2024-07-26 16:50:13.613311
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
print(type(date))



path = './_save/keras32_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k32_04',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
          verbose=1, validation_split=0.3,
          callbacks=[es, mcp],
          )
end = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0725_1545.csv")

print('로스 : ', loss)
print("r2스코어 : ", r2)
print('걸린시간 : ', round(end - start, 2), "초")



# 로스 :  1581.5755615234375
# 걸린시간 :  5.97 초


# Dropout
# 로스 :  1620.7923583984375
# r2스코어 :  0.7439298559440584
# 걸린시간 :  6.0 초
