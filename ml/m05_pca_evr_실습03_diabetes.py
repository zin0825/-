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
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = load_diabetes()
print(datasets)
# print(datasets.DESCR)   # describe 확인 
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape)   # (442, 10)
print(y.shape)   # (442,)

# exit()

pca = PCA(n_components=10)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 8
# print(np.argmax(cumsum >= 0.99) +1)  # 8
# print(np.argmax(cumsum >= 0.999) +1) # 9
# print(np.argmax(cumsum >= 1.0) +1)   # 10

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=9)


n = [8, 8, 8, 10]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=n[i]))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))


    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    start = time.time()

    es = EarlyStopping(monitor= 'val_loss',
                    mode= 'min',
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

    path = './_save/m05_03/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_03_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train1, y_train, epochs=100, batch_size=3, 
                    verbose=0, validation_split=0.3,
                    callbacks=[es, mcp])

    end = time.time()


    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')

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

# pca
# 결과 PCA : 8
# acc :  0.0
# 걸린 시간 :  4.26 초
# ===========================
# 결과 PCA : 8
# acc :  0.0
# 걸린 시간 :  5.49 초
# ===========================
# 결과 PCA : 8
# acc :  0.0
# 걸린 시간 :  4.17 초
# ===========================
# 결과 PCA : 10
# acc :  0.0
# 걸린 시간 :  4.23 초
# ===========================
