import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint



#1. 데이터
path = './_data/dacon/따릉이/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

# print(train_csv.shape)   # (1459, 10)
# print(test_csv.shape)   # (715, 9)
# print(submission_csv.shape)   # (715, 1)

# print(train_csv.columns)

# print(train_csv.info())
# print(train_csv.isna().sum())

train_csv = train_csv.dropna()
# print(train_csv.isna().sum())
# print(train_csv)   # [1328 rows x 10 columns]

# print(train_csv.isna().sum())
# print(train_csv.info())

# print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
# print(x)   # [1328 rows x 9 columns]

y = train_csv['count']
# print(y)

# print(x.shape, y.shape)   # (1328, 9)(1328,)


# exit()

pca = PCA(n_components=9)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 1
# print(np.argmax(cumsum >= 0.99) +1)  # 1
# print(np.argmax(cumsum >= 0.999) +1) # 3
# print(np.argmax(cumsum >= 1.0) +1)   # 9

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=5757)


n = [1, 1, 3, 9]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=n[i]))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(79, activation='relu'))
    model.add(Dense(50, activation='relu'))
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

    path = './_save/m05_04/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_04_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train1, y_train, epochs=100, batch_size=32, 
            verbose=0, validation_split=0.3,
            callbacks=[es, mcp])

    end = time.time()


    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')

# y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)

# submission_csv['count'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)

# submission_csv.to_csv(path + "submission_0725_1545.csv")

# print('로스 : ', loss)
# print('걸린시간 : ', round(end - start, 2), "초")

# 로스 :  2208.22509765625
# 걸린시간 :  4.86 초

# 로스 :  2196.84375
# 걸린시간 :  5.29 초

# 스켈링
# 로스 :  1465.239501953125
# 걸린시간 :  4.88 초

# 로스 :  1479.632080078125
# 걸린시간 :  4.93 초

# pca
# 결과 PCA : 1
# acc :  0.0
# 걸린 시간 :  5.09 초
# ===========================
# 결과 PCA : 1
# acc :  0.0
# 걸린 시간 :  1.75 초
# ===========================
# 결과 PCA : 3
# acc :  0.0
# 걸린 시간 :  3.75 초
# ===========================
# 결과 PCA : 9
# acc :  0.0
# 걸린 시간 :  2.07 초
# ===========================

