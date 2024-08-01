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
                                                   random_state=457)

print(x_train.shape, y_train.shape)   # (49502, 93) (49502,)
print(x_test.shape, y_test.shape)   # (12376, 93) (12376,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler 
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

test_csv = scaler.transform(test_csv)

print(x_train)   
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.4615384615384617



#2. 모델구성
model = Sequential()
model.add(Dense(200, activation='relu', input_dim=93))
model.add(Dense(200, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='softmax'))

#3. 컴파일,  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, 
                   verbose=1,  restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath='./_save/keras30_mcp/keras30_13_kaggle_otto.hdf5')


hist = model.fit(x_train, y_train, epochs=700, batch_size=2232, 
                 verbose=1,  
                 validation_split=0.3,
                 callbacks=[es, mcp])
end = time.time()

model.save('./_save/keras30_mcp/keras30_13_kaggle_otto_save.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

sample_csv[['Class_1',	'Class_2',	'Class_3',	'Class_4',	'Class_5',	
            'Class_6',	'Class_7',	'Class_8',	'Class_9']] = y_submit

sample_csv.to_csv(path + "sampleSubmission_0725_1623.csv")

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

# 스켈링
# acc score :  0.7462831286360698
# 걸린 시간 :  7.19 초
# 로스 :  [0.5738803744316101, 0.7899159789085388]

# acc score :  0.7429702650290886
# 걸린 시간 :  48.89 초
# 로스 :  [0.6805111169815063, 0.7722204327583313]


# acc score :  0.7744828700711054
# 걸린 시간 :  112.46 초
# 로스 :  [2.5615627765655518, 0.7759373188018799]


# save
# acc score :  0.7463639301874596
# 걸린 시간 :  8.38 초
# 로스 :  [0.5989190340042114, 0.779411792755127]