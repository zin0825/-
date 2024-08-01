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
# scaler = MinMaxScaler()
scaler = MinMaxScaler()

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

from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss',
                   mode= 'min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras30_mcp/keras30_03_diabetes.hdf5')


hist = model.fit(x_train, y_train, epochs=100, batch_size=3, 
          verbose=1, validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()


model.save('./_save/keras30_mcp/keras30_03_diabetes_save.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


# 로스 :  2368.745849609375
# r2스코어 :  0.5647267605952928