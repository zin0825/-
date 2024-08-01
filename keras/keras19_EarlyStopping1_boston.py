import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import rc

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # Description 속성을 이용해서 데이터셋의 정보를 확인
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=3, 
                                                    train_size=0.7)
#x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)

print(x)
print(y)
print(x.shape, y.shape) 

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=13))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation= 'linear'))


#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',   # 기준
    mode = 'min',   # 최소값을 찾을거야. 모르면 auto 알고있으면 min / 자동으로 loss는 최소값, acc, r2는 최대값
    patience=10,   # 참을성, 훈련동안 갱신되지 않으면 훈련을 끝냄
    restore_best_weights=True,   # y=wx + b의 최종 가중치 어쩔 땐 안쓰는게 좋을 수도 있음
)   # 최소값 지점을 최종 가중치로 잡음. 쓰지않으면 훈련이 끝난 지점의 가중치를 잡음

hist = model.fit(x_train, y_train, epochs=100, batch_size = 32, 
                 verbose=1, validation_split=0.3,
                 callbacks=[es]
                 )   # [] = 리스트 / 두개이상은 리스트 = 나중에 또 친구 나오겠다
end = time.time()

# 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('로스 : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

print("걸린 시간 : ", round(end - start,2),'초')

print("=================== hist ==================")
print(hist)

print("================ hist.history =============")
print(hist.history)

print("================ loss =============")
print(hist.history['loss'])
print("================ val_loss =============")
print(hist.history['val_loss'])
print("==================================================")


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'   # 한글 깨짐, 폰트 적용

plt.figure(figsize=(9,6))   # 9 x 6 사이즈 
plt.plot(hist.history['loss'],c='red', label='loss',)
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
plt.legend(loc='upper right')
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()


# 로스 :  27.968557357788086
# r2 score :  0.6443942603971708
# 걸린 시간 :  1.74 초

# 로스 :  23.992708206176758
# r2 score :  0.6949451313278959
# 걸린 시간 :  1.64 초