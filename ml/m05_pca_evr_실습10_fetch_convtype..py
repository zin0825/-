from sklearn.datasets import fetch_covtype

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (581012, 54) (581012,)

# exit()

pca = PCA(n_components=54)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 2
# print(np.argmax(cumsum >= 0.99) +1)  # 4
# print(np.argmax(cumsum >= 0.999) +1) # 5
# print(np.argmax(cumsum >= 1.0) +1)   # 52

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)   # y는 예스 아니고 y

n = [2, 4, 5, 52]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)



    #2. 모델구성
    model = Sequential()
    model.add(Dense(180, activation='relu', input_dim=n[i]))
    model.add(Dense(90))
    model.add(Dense(46))
    model.add(Dense(6))
    model.add(Dense(7, activation='softmax'))


    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    start = time.time()

    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=10, restore_best_weights=True)

    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    import datetime
    date = datetime.datetime.now()
    # print(date)    
    # print(type(date))  
    date = date.strftime("%m%d_%H%M")
    # print(date)     
    # print(type(date))  

    path = './_save/m05_10/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_10_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train, y_train, epochs=10, batch_size=2586, 
                    verbose=0, validation_batch_size=0.3,
                    callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')



# 케라스
# acc score :  0.48645485525455234
# 걸린 시간 :  8.08 초
# 로스 :  [0.93593430519104, 0.5576055645942688]

# acc score :  0.6604419813431551
# 걸린 시간 :  7.26 초
# 로스 :  [1.1363941431045532, 0.6739699244499207]

# ra- 3086, ep- 500, ba- 3086
# acc score :  0.8239991738666483
# 걸린 시간 :  483.44 초
# 로스 :  [0.401950865983963, 0.8311246037483215]

# ra- 3086, ep- 700, ba- 3086
# acc score :  0.8350142852225396
# 걸린 시간 :  675.04 초
# 로스 :  [0.37904050946235657, 0.8404185771942139]


# 사이킷런
# acc score :  0.7697497504388834
# 걸린 시간 :  61.56 초
# 로스 :  [0.5086667537689209, 0.7800419926643372]

# acc score :  0.7845168841003752
# 걸린 시간 :  57.34 초
# 로스 :  [0.47943201661109924, 0.7985267043113708]

# 스켈링
# acc score :  0.7627620391724897
# 걸린 시간 :  5.93 초
# 로스 :  [0.519395649433136, 0.777374267578125]