# 89이상
# 다중분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# tf274gpu로 버전 변경
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


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



# #2. 모델구성
# model = Sequential()
# model.add(Dense(200, activation='relu', input_dim=93))
# model.add(Dropout(0.6))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(130, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(130, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(9, activation='softmax'))


#2-2. 모델구성(함수형)
input1 = Input(shape=(93,))
dense1 = Dense(200, name='ys1')(input1)
drop1 = Dropout(0.6)(dense1)
dense2 = Dense(200, name='ys2')(drop1)
drop2 = Dropout(0.6)(dense2)
dense3 = Dense(130, name='ys3')(drop2)
drop3 = Dropout(0.6)(dense3)
dense4 = Dense(130, name='ys4')(drop3)
drop4 = Dropout(0.6)(dense4)
dense5 = Dense(100, name='ys5')(drop4)
dense6 = Dense(100, name='ys6')(dense5)
dense7 = Dense(50, name='ys7')(dense6)
dense8 = Dense(40, name='ys8')(dense7)
dense9 = Dense(10, name='ys9')(dense8)
dense10 = Dense(10, name='ys10')(dense9)
output1 = Dense(9, activation='softmax')(dense10)
model = Model(inputs=input1, outputs=output1)
model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 93)]              0

#  ys1 (Dense)                 (None, 200)               18800

#  dropout (Dropout)           (None, 200)               0

#  ys2 (Dense)                 (None, 200)               40200

#  dropout_1 (Dropout)         (None, 200)               0

#  ys3 (Dense)                 (None, 130)               26130

#  dropout_2 (Dropout)         (None, 130)               0

#  ys4 (Dense)                 (None, 130)               17030

#  dropout_3 (Dropout)         (None, 130)               0

#  ys5 (Dense)                 (None, 100)               13100

#  ys6 (Dense)                 (None, 100)               10100

#  ys7 (Dense)                 (None, 50)                5050

#  ys8 (Dense)                 (None, 40)                2040

#  ys9 (Dense)                 (None, 10)                410

#  ys10 (Dense)                (None, 10)                110

#  dense (Dense)               (None, 9)                 99

# =================================================================
# Total params: 133,069
# Trainable params: 133,069
# Non-trainable params: 0


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



if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxxx")
    


# acc score :  0.7334356819650937
# 걸린 시간 :  4.17 초
# 로스 :  [0.5940346717834473, 0.7771493196487427]

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


# save
# acc score :  0.7463639301874596
# 걸린 시간 :  8.38 초
# 로스 :  [0.5989190340042114, 0.779411792755127]


# acc score :  0.6813994828700711
# 걸린 시간 :  28.26 초
# 로스 :  [0.6858252882957458, 0.744182288646698]
# 쥐피유 없다! xxxxx

# acc score :  0.6813186813186813
# 걸린 시간 :  15.77 초
# 로스 :  [0.6869246959686279, 0.7440207004547119]
# 쥐피유 돈다!!!