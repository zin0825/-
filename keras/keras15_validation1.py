#08_1 카피 (with 14)


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6])   # validation 훈련에 영향이 조오오금 미침 / 훈련은 여기서
y_train = np.array([1,2,3,4,5,6])   
# 임의로 정한 숫자임 비율이 중요한게 아님 
# 3등분으로 나눠서 훈련, 검증, 테스트 진행
# 훈련 , 검증, 훈련, 검증으로 트레인과 validation을 반복 -> evaluate

x_val = np.array([7,8])   # 8:2 /트레인에서 발리데이션 검증은 75:25 나눔
y_val = np.array([7,8])   # 트레인과 테스트와 나눴음

x_test = np.array([9,10])
y_test = np.array([9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val),   # 이 파일에서 요놈만 추가 / (x_val, y_val)로 검증해
          )   
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


# 6/6 [==============================] - 0s 2ms/step - loss: 0.0270 - val_loss: 0.0817   # val 이 생김
# +++++++++++++++++++++++++++++++++++++++++
# 1/1 [==============================] - 0s 34ms/step - loss: 0.2104
# 로스 :  0.2103712558746338
# [11]의 예측값 :  [[10.412885]]   # val보다 로스가 안좋고 평가(evaluate)가 더 안좋음.