# 09_2 카피
# 검색 R²

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9, 6,17,23,21,20]) 

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런

from sklearn.model_selection import train_test_split   # 키워드를 _로 표시

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    train_size=0.7,   # 70%   # 디폴트값 0.75
                                    # test_size=0.4   # 오버된다
                                    # shuffle=True,   # 랜덤하게 섞겠다  디폴트 트루
                                    random_state=104,   # 훈련할 데이터 섞였다, 랜덤값을 고정해주는것
                                          # 셔플과 랜덤은 같이 사용해야 한다. 셔플을 F로 한다면 랜덤을 쓸 필요없다
                                    )  # 드래그 Tab 오른쪽, 쉬프트 + Tab 왼쪽

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)


y_predict = model.predict(x_test)  
from sklearn.metrics import r2_score, mean_squared_error    
# 선형회귀 분석, 회귀 모델의 적합도를 나타냄
# https://sjkoding.tistory.com/66
r2 = r2_score(y_test,y_predict)   
# 로스를 테스트한 테이터와 원값 데이터를 평가함
# 1에 가까울수록 높은 성능
print("r2스코어 : ", r2)


# ===================
def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))   # np.sqrt 루트를 씌우다
rmse = RMSE(y_test, y_predict)   # 내가 사용할 함수 명을 사용
print("RMSE : ", rmse)



# random_state=104
# 로스 :  9.651159286499023
# r2스코어 :  0.6900609106489045

# random_state=11
# 로스 :  7.66412878036499
# r2스코어 :  0.7276322279812748



# rmse
# 로스 :  9.681751251220703
# r2스코어 :  0.689078453418344
# RMSE :  3.1115512999415063
