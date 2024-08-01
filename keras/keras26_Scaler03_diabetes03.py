import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_diabetes()
print(datasets)
print(datasets.DESCR)   # describe 확인 
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)

print(x)
print(y)
print(x.shape, y.shape)   # (442, 10) (442,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0  # 이건 랜덤값에 따라 0이 나올수도 -가 나올 수도 있다. 
# 랜덤 33 - 0.0 1.0  랜덤 9 - -어쩌구



#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=10))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor= 'val_loss',
                   mode= 'min',
                   patience=10,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=3, 
                 verbose=1, validation_split=0.3)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

# 랜덤 33
# 로스 :  2581.506103515625
# r2스코어 :  0.557094199511507

# 로스 :  2633.047119140625
# r2스코어 :  0.5482513621347966

# 스켈링
# 로스 :  2642.084228515625
# r2스코어 :  0.5467008969506433

# 로스 :  2658.826904296875
# r2스코어 :  0.5438283941746138

# 랜덤 9
# 로스 :  2245.917724609375
# r2스코어 :  0.5872972834016805

# 스켈링
# 로스 :  2226.10400390625
# r2스코어 :  0.5909382253854453

# 로스 :  2123.247802734375
# r2스코어 :  0.6098386941406617


# StandardScaler 
# 로스 :  3213.656494140625
# r2스코어 :  0.4094686539530493


# MaxAbsScaler
# 로스 :  2947.380859375
# r2스코어 :  0.45839863513406764

# 로스 :  2671.080322265625
# r2스코어 :  0.5091707481564929


# RobustScaler
# 로스 :  2701.5595703125
# r2스코어 :  0.5035699731118626

# 로스 :  2939.347412109375
# r2스코어 :  0.45987488566746426