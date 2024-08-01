import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.65, 
                                                    shuffle=True, 
                                                    random_state=133)

print(x_train, y_train)
print(x_test, y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1,
          # validation_data=(x_val, y_val),
          validation_split=0.3)   # 트레인 데이터의 30%을 발리데이션으로 쓰겠다


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 scoer : ', r2)

results = model.predict([18])
print('[18]의 예측값 : ', results)


# 로스 :  2.3462523035533422e-09
# r2 scoer :  0.9999999998945505
# [18]의 예측값 :  [[17.999918]]

# 로스 :  0.0006805545999668539
# r2 scoer :  0.9999694132771874
# [18]의 예측값 :  [[17.963453]]