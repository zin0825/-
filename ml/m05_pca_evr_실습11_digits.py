import numpy as np
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


from sklearn.datasets import load_digits   # digits 숫자
import pandas as pd


#1. 데이터
x, y = load_digits(return_X_y=True)   # x와 y로 바로 반환해줌
print(x)
print(y)
print(x.shape, y.shape)   # (1797, 64) (1797,)   이미지는 0에서 225의 숫자를 부여함 225가 가장 진함 놈

# exit()

pca = PCA(n_components=64)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 29
# print(np.argmax(cumsum >= 0.99) +1)  # 41
# print(np.argmax(cumsum >= 0.999) +1) # 49
# print(np.argmax(cumsum) +1)   # 61

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=3308,
                                                    stratify=y)

n = [29, 41, 49]
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
    model.add(Dense(10, activation='softmax'))


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

    path = './_save/m05_11/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_11_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train, y_train, epochs=180, batch_size=3086, 
                    verbose=0, validation_batch_size=0.3,
                    callbacks=[es, mcp])

    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')



# # 케라스
# # acc score :  0.9777777777777777
# # 걸린 시간 :  0.87 초
# # 로스 :  [0.11199933290481567, 0.9833333492279053]

# # 판다스
# # acc score :  0.9333333333333333
# # 걸린 시간 :  0.88 초
# # 로스 :  [0.18980160355567932, 0.9388889074325562]

# # 사이킷런
# # acc score :  0.9833333333333333
# # 걸린 시간 :  0.8 초
# # 로스 :  [0.0742197334766388, 0.9833333492279053]

# acc score :  0.9888888888888889
# 걸린 시간 :  1.25 초
# 로스 :  [0.05170276015996933, 0.9888888597488403]

# 스켈링
# acc score :  0.9555555555555556
# 걸린 시간 :  1.14 초
# 로스 :  [0.14160378277301788, 0.9611111283302307]

# pca


