import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런

from sklearn.model_selection import train_test_split   # 키워드를 _로 표시
# 머신러인 분석 라이브러리

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False, random_state=0)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

# [1 2 3 4 5 6 7]
# [1 2 3 4 5 6 7]
# [ 8  9 10]
# [ 8  9 10]


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    train_size=0.7,   # 70%   # 디폴트값 0.75
                                    #test_size=0.4   # 오버된다 # 디폴트값 0.25
                                    #shuffle=True,   # 랜덤하게 섞겠다  디폴트 트루
                                    random_state=1004,   # 훈련할 데이터 섞였다, 랜덤값을 고정해주는것
                                          # 셔플과 랜덤은 같이 사용해야 한다. 셔플을 False로 한다면 랜덤을 쓸 필요없다
                                    )  # 드래그 Tab 오른쪽, 쉬프트 + 드래그 왼쪽

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)


# shuffle=True
# random_state=123, 
# x_train :  [ 6  9  4  2  7 10  3]
# x_test :  [5 1 8]
# y_train :  [ 6  9  4  2  7 10  3]
# y_test :  [5 1 8]


# x_train :  [ 8  7  2  6 10  3  4]   # 셔플 주석 - 디폴트가 트루가 된다
# x_test :  [5 9 1]
# y_train :  [ 8  7  2  6 10  3  4]
# y_test :  [5 9 1]



# aaa = 3
#   bbb = 4   # 줄 간격이 바뀌면 꼬붕

def train_test_spilt(a, b):
   a = a+b
   return x_train, x_test, y_train, y_test # rain_test_spilt 리턴 반환된다.



# train_size=0.7,
# random_state=1004, 
# x_train :  [10  2  5  4  8  6  3]
# x_test :  [1 7 9]
# y_train :  [10  2  5  4  8  6  3]
# y_test :  [1 7 9]

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
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)



# x_train :  [10  2  5  4  8  6  3]
# x_test :  [1 7 9]
# y_train :  [10  2  5  4  8  6  3]
# y_test :  [1 7 9]
# 로스 :  0.03498458117246628
# [11]의 예측값 :  [[10.7456665]]