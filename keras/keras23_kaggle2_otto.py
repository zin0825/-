# 89이상
# 다중분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
path = ".\\_data\\keggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)   # (61878, 94)
# print(test_csv.shape)   # (144368, 93)
# print(sample_csv.shape)   # (144368, 9)

# print(train_csv.isnull().sum())
# print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])
print(train_csv.shape)

x = train_csv.drop(['target'], axis=1)
print(x.shape)   # [(61878, 93)

y = train_csv['target']
print(y.shape)   # (61878,)

y = pd.get_dummies(y)  
print(y)   # [61878 rows x 9 columns]


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   train_size=0.8,
                                                   shuffle=True,
                                                   random_state=5757)

print(x_train.shape, y_train.shape)   # (49502, 93) (49502,)
print(x_test.shape, y_test.shape)   # (12376, 93) (12376,)

#2. 모델구성
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=93))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(9, activation='softmax'))

#3. 컴파일,  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=2200, 
                 validation_batch_size=0.3,
                 callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

sample_csv[['Class_1',	'Class_2',	'Class_3',	'Class_4',	'Class_5',	
            'Class_6',	'Class_7',	'Class_8',	'Class_9']] = y_submit

sample_csv.to_csv(path + "sampleSubmission_0725_1236.csv")

accuracy_score = accuracy_score(y_test, y_pred)

print('acc score : ', accuracy_score)
print('걸린 시간 : ', round(end - start, 2), "초")
print('로스 : ', loss)

# acc score :  0.7334356819650937
# 걸린 시간 :  4.17 초
# 로스 :  [0.5940346717834473, 0.7771493196487427]

# acc score :  0.7805429864253394
# 걸린 시간 :  370.3 초
# 로스 :  [2.9506893157958984, 0.7813510298728943]

# acc score :  0.7618778280542986
# 걸린 시간 :  103.16 초
# 로스 :  [0.762648344039917, 0.7828054428100586]

# acc score :  0.7506464124111183
# 걸린 시간 :  7.58 초
# 로스 :  [0.5755793452262878, 0.7859566807746887]