import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)   # 자세히 보기
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (150, 4) (150,)

print(y)   # 셔플을 안 할 경우 밑 숫자에서 작살남
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

#### 맹그러봐!!!! ####
# 150 -> 150, 3
# 판다스, 케라스, 사이킷런


# # 사이킷런
# from sklearn.preprocessing import OneHotEncoder

# oh = OneHotEncoder(sparse=False)
# y = oh.fit_transform(y.reshape(-1,1))   # reshap 형태 바꿔준다
# print(y.shape)   # (150, 3)
# # 로스 :  [0.046233925968408585, 1.0]

# # 케라스
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)
# # 로스 :  [0.011925656348466873, 1.0]

# # 판다스 
# y = pd.DataFrame(y)
# y = pd.get_dummies(y[0])   # 0을 붙이는것은 첫번째를 구하기 때문 
# print(y)   # (150, 3)
# # 로스 :  [0.03643076494336128, 1.0]


# reshape 데이터의 값이 바뀌면 안된다. 데이터의 순서가 바뀌면 안된다

# print("======================================")
# # 선생님 코드
# # 원핫 1. 케라스
# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)   # (150, 3)
# # 로스 :  [0.011925656348466873, 1.0]


print("======================================")
# 원핫 2. 판다스 
y_ohe2 = pd.get_dummies(y)  
print(y_ohe2)
print(y_ohe2.shape)   # (150, 3)
# 로스 :  [0.03643076494336128, 1.0]


# print("======================================")
# # 원핫 3. 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y_ohe3 = ohe.fit_transform(y_ohe3)  # reshap 형태 바꿔준다

# print(y_ohe3)   # (150, 3)
# # 로스 :  [0.046233925968408585, 1.0]



# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical

# t = Tokenizer()
# t.fit_on_texts([text])
# print(t.word_index)




x_train, x_test, y_train, y_test = train_test_split(x, y_ohe2,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)




print(x.shape)
print(y.shape)
# (150, 4)
# (150,)
# -> (150, 3)로 변경 / 벡터 하나는 하나의 컬럼, 열, 피쳐, 특성



#2. 모델구성
model = Sequential()
model.add(Dense(180, activation='relu', input_dim=4))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras. callbacks import EarlyStopping

# model.compile(loss='mae', optimizer='adam', metrics=['acc'])
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=10, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                 validation_batch_size=0.2,
                 callbacks=[es])
end = time.time()

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


# acc score :  0.9333333333333333
# 걸린 시간 :  1.69 초
# 로스 :  [0.019203782081604004, 0.8666666746139526]

# acc score :  1.0
# 걸린 시간 :  3.48 초
# 로스 :  [0.04420299828052521, 1.0]
