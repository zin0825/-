#08_1 카피


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=4)   
# epoch가 1개 처리하는데 0.1초만에 끝나기에 사람이 볼 수 없음 그래서 사람이 볼 수 있게 딜레이 시킴  
# verbose=0은 바로 결과 나옴, 1는 디폴트,  2 진행바 빼고, 3, 4....는 epoch만 나옴

# verbose=0 : 침묵
# verbose=1 : 디폴트
# verbose=2 : 프로그래스바 삭제
# verbose=나머지 : 에포만 나온다.
 



#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)


# 7/7 [==============================] - 0s 332us/step - loss: 0.0153    # 훈련
# Epoch 100/100
# 7/7 [==============================] - 0s 332us/step - loss: 0.0144
# +++++++++++++++++++++++++++++++++++++++++
# 1/1 [==============================] - 0s 58ms/step - loss: 0.0592    # 평가
# 로스 :  0.059226710349321365
# [11]의 예측값 :  [[10.650603]]