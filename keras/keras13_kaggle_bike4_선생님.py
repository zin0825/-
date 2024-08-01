# 기존 캐글 데이터에서
# 1. train_csv의  y를 casual과 register로 잡는다.
#    그래서 훈련을 해서 test_csv의 casual과 register를 predict한다.
   
# 2. test_csv에 casual과 register 컬럼을 합쳐

# 3. train_csv에 y를 count로 잡는다.

# 4. 전체 훈련

# 5. test_csv 예측해서  submission 에 붙여!!!


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\keggle\\bike-sharing-demand\\'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test_csv.shape)
print(sampleSubmission_csv.shape)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())  

print(train_csv.isna().sum())   # 0
print(train_csv.isnull().sum())   # 0

print(test_csv.isna().sum())   # 0
print(test_csv.isnull().sum())   # 0

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)

y = train_csv[['casual', 'registered']]   # 리스트가 2개 이상이니까 [[]]
print(y.shape)   # (10886, 2)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    random_state=100)

# 트레인과 테스트를 분리했어 눈으로 확인해야해 모델.이벨류 데이터
# 훈련->검증 x @@@ 발리데이션
# 발리데이션을 #1에서 말고 #3에서 스플릿트로 비율을 나눈다
# 트레인 발리데이션 테스트로 분류

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)  


#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=8))
model.add(Dense(95, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


print(test_csv.shape)   # (6493, 8)
   
y_submit = model.predict(test_csv)
print(test_csv.shape)
print(y_submit.shape)   # (6493, 2)   집어넣으려 보니까 (6493, 8) (6493, 2) 두개 합쳐야해


print("test_csv타입 : ", type(test_csv))
print("y_submit타입 : ", type(y_submit))
# test_csv타입 :  <class 'pandas.core.frame.DataFrame'>
# y_submit타입 :  <class 'numpy.ndarray'>   # 넘파이를 판다스에 넣어야해 컬럼

test2_csv = test_csv
print(test2_csv.shape)   # (6493, 8)

test2_csv[['casual','registered']] = y_submit
# 테스트2에 캐주얼 컬럼을 만들겠다   y_submit 얘도 리스트니까 2개 만들어
print(test2_csv)   # [6493 rows x 10 columns]

test2_csv.to_csv(path + "test2.csv")   


# 모델 최소 2번은 돌려야함
# 속도 높일라면 배치를 늘리면 돼
# 버보스 디폴트 1 뵈기싫으면 0
# 버보스 2박3일 일땐 언제끝날지 모르니까 표시할것





# 로스 :  8651.05859375








