from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

from tensorflow.keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)

print(x.shape)
print(y.shape)
# (178, 13)
# (178,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.05128205128205132 0.9896551724137932





#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=13))
model.add(Dense(90, activation='relu'))
model.add(Dense(46, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, 
                   verbose=1,
                   restore_best_weights=True)
 
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras30_mcp/keras30_09_wine.hdf5')
 
 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()

model.save('./_save/keras30_mcp/keras30_09_wine_save.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("ACC : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print(y_predict)


print("acc score : ", accuracy_score)
print("걸린 시간 : ", round(end - start, 2), "초")
print("로스 : ", loss)


# acc score :  0.9444444444444444
# 걸린 시간 :  1.92 초
# 로스 :  [0.08138372004032135, 0.9444444179534912]
