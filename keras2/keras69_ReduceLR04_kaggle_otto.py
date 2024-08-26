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
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337)   # 값 고정
np.random.seed(337)


#1. 데이터
path = ".\\_data\\kaggle\\otto-group-product-classification-challenge\\"

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
print(train_csv.shape)   # (61878, 94)

x = train_csv.drop(['target'], axis=1)
print(x.shape)   # [(61878, 93)

y = train_csv['target']
print(y.shape)   # (61878,)


# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,
                                                    random_state=337,
                                                    )

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
results = []

########## for 문 #############

for learning_rate in lr:

    #2. 모델구성
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=x_train.shape[1]))
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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))   
    
    model.fit(x_train, y_train, 
              epochs=700, batch_size=2232,
              verbose=0, validation_split=0.2)


    #4. 평가, 예측
    print('=================1. 기본 출력 =================')
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))   # 로스값이 {}에 쏙 들어간다

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


# sample_csv[['Class_1',	'Class_2',	'Class_3',	'Class_4',	'Class_5',	
#             'Class_6',	'Class_7',	'Class_8',	'Class_9']] = y_submit

# sample_csv.to_csv(path + "sampleSubmission_0823_1132.csv")


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

# learning_rate

