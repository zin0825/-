import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9, 6,17,23,21,20]) 

# 맹그러서 그려봐!!!
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    train_size=0.7,   # 70%   # 디폴트값 0.75
                                    # test_size=0.4   # 오버된다
                                    # shuffle=True,   # 랜덤하게 섞겠다  디폴트 트루
                                    # random_state=1004,   # 훈련할 데이터 섞였다, 랜덤값을 고정해주는것
                                    # 셔플과 랜덤은 같이 사용해야 한다. 셔플을 F로 한다면 랜덤을 쓸 필요없다
                                    )  # 드래그 Tab 오른쪽
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
results = model.predict([x])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, results, color='red')
plt.show()