import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = load_breast_cancer()
# print(datasets)   # 한개의 행[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01, 1.189e-01]
#  y 값 array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...

# print(datasets.DESCR)   
# # Number of Instances: 569 행
# # Attributes: 30  속성, 열
# # Missing Attribute Values: None 결측치 있냐
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)   # (569, 30) (569,)

# y는 넘파이 데이터에서 암이 반절이 걸렸다. 0,1,2, 다중 분류일 경우 다 확인해야하는데
# y의 라벨값을 묻는것이 있다.
# 넘파이에서 y가 0과 1의 종류

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

print(np.argmax(cumsum >= 0.95) +1)  # 1
print(np.argmax(cumsum >= 0.99) +1)  # 2
print(np.argmax(cumsum >= 0.999) +1) # 3
print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=99, 
                                                    train_size=0.9)


n = [1, 1, 1, 8]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=n[i]))   
    model.add(Dense(16,activation='relu'))   # 통상적으로 2의 배수, 이진법이 잘 먹힘
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='sigmoid'))   # 중간에 시그모이드 써도 됨
    model.add(Dense(16, activation='relu'))   # 렐루, 시그모이드 뭐가 좋아? 해봐야함
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 분류에서  마지막 레이어 엑큐베이션은 리니어,
    # 이진 분류에서는 마지막 엑티베이션은 시그마 = 0에서 1사이의 값을 바꿔준다


    #3. 컴파일, 훈련
    model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])   #  accuracy, mse

    start = time.time()

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(
        monitor = 'val_loss',   # 최소값을 찾을거야
        mode = 'min',   # 모르면 auto 알고있으니가 min
        patience=20,   # 참을성
        restore_best_weights=True,   # y=wx + b의 최종 가중치 어쩔 땐 안쓰는게 좋을 수도 있음
    )   # 얼리스탑핑을 정리하는게 끝


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    import datetime
    date = datetime.datetime.now()
    # print(date)    
    # print(type(date))  
    date = date.strftime("%m%d_%H%M")
    # print(date)     
    # print(type(date))  

    path = './_save/m05_06/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_06_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train1, y_train, epochs=1000, batch_size = 8, 
                    verbose=0, validation_split=0.2,
                    callbacks=[es, mcp]
                    )   # [] = 리스트 / 두개이상은 리스트 = 나중에 또 친구 나오겠다

    end = time.time()


    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')





# 로스 :  [0.3625730872154236, 0.6374269127845764]   # 로스 두번째 값이 엑큐러시
# [[1.]   # 모델이 구리면 에러남 / print(y_pred)로 확인
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]]

# loss: 0.2326 - val_loss: 0.2431 -> mse임


# 로스 :  0.3625730872154236
# ACC :  0.637

# 32,32,22,16,1
# 로스 :  0.05437751114368439
# ACC :  0.936

# 32,16,16,16,1
# 로스 :  0.06999842822551727
# ACC :  0.918
# acc_score :  1.0
# 걸리시간 :  1.03 초


# 스켈링
# acc_score :  0.9473684210526315
# 걸리시간 :  2.41 초
# 로스 :  0.03002031147480011

# pca
# 결과 PCA : 1
# acc :  0.9122806787490845
# 걸린 시간 :  24.51 초
# ===========================
# 결과 PCA : 1
# acc :  0.9122806787490845
# 걸린 시간 :  22.77 초
# ===========================
# 결과 PCA : 1
# acc :  0.9122806787490845
# 걸린 시간 :  21.02 초
# ===========================
# 결과 PCA : 8
# acc :  0.9824561476707458
# 걸린 시간 :  11.88 초
# ===========================