"""
01. 보스톤
02. california
03. diabetes
04. dacon_ddarung
05. kaggle_bike

06_cancer
07_dacon_diabetes
08_kaggle_bank
09_wine
10_fetch_covtpe
11_digits
"""


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from matplotlib import re
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape)   # (20640, 8)
# print(y.shape)   # (20640,)

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 1
# print(np.argmax(cumsum >= 1.0) +1)   # 8

# exit()

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=20)

n = [1, 1, 1, 8]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(30, activation='relu', input_dim=n[i]))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))


    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    start = time.time()

    from  tensorflow.keras. callbacks import EarlyStopping
    es = EarlyStopping(monitor = 'val_loss',
                    mode= 'min',
                    verbose=0,
                    patience=10,
                    restore_best_weights=True)
    
    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    import datetime
    date = datetime.datetime.now()
    # print(date)    
    # print(type(date))  
    date = date.strftime("%m%d_%H%M")
    # print(date)     
    # print(type(date))  

    path = './_save/m05_02/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_02_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train1, y_train, epochs=100, batch_size=64, 
            verbose=0, validation_split=0.3,
            callbacks=[es, mcp])

    end = time.time()



    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')




# 로스 :  0.7055579423904419
# r2스코어 :  0.4942015585919515
# 걸린시간 :  3.92 초

# 로스 :  0.6498311161994934
# r2스코어 :  0.5341510484911183
# 걸린시간 :  5.54 초

# 스켈링
# 로스 :  0.3118913173675537
# r2스코어 :  0.7764122651218553
# 걸린시간 :  16.23 초

# 로스 :  0.3162597119808197
# r2스코어 :  0.7732805969677623
# 걸린시간 :  16.02 초

# pca
# 결과 PCA : 1
# acc :  0.0007267441833391786
# 걸린 시간 :  12.1 초
# ===========================
# 결과 PCA : 1
# acc :  0.0007267441833391786
# 걸린 시간 :  19.27 초
# ===========================
# 결과 PCA : 1
# acc :  0.0007267441833391786
# 걸린 시간 :  12.84 초
# ===========================
# 결과 PCA : 8
# acc :  0.0007267441833391786
# 걸린 시간 :  19.41 초
# ===========================