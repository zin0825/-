from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (178, 13) (178,)
print(np.unique(y, return_counts=True)) 
# exit()


y = pd.get_dummies(y)
print(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186,
                                                    stratify=y)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)  

# exit()

print(x_train.shape, x_test.shape)   # (160, 13) (18, 13)

# exit()


pca = PCA(n_components=x_train.shape[1])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 10
# print(np.argmax(cumsum >= 0.99) +1)  # 12
# print(np.argmax(cumsum >= 0.999) +1) # 13
# print(np.argmax(cumsum) +1)   # 13

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1186)

n = [10, 12, 13, 13]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델 구성
    model = Sequential()
    model.add(Dense(150, activation='relu', input_dim=n[i]))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(46, activation='relu'))
    model.add(Dense(22, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))


    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    start = time.time()

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

    path = './_save/m05_09/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_09_date_', str(i+1),'_', date, '_epo_', filename])  


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


# acc score :  0.9444444444444444
# 걸린 시간 :  3.56 초
# 로스 :  [0.1882103979587555, 0.9444444179534912]

# acc score :  0.9444444444444444
# 걸린 시간 :  3.63 초
# 로스 :  [0.10632947087287903, 0.9444444179534912]

# acc score :  0.9444444444444444
# 걸린 시간 :  3.67 초
# 로스 :  [0.11336535960435867, 0.9444444179534912]

# 스켈링
# acc score :  0.9444444444444444
# 걸린 시간 :  5.71 초
# 로스 :  [0.2595745921134949, 0.9444444179534912]

# pca
# 결과 PCA : 10
# acc :  0.8888888955116272
# 걸린 시간 :  5.81 초
# ===========================
# 결과 PCA : 12
# acc :  0.8888888955116272
# 걸린 시간 :  9.95 초
# ===========================
# 결과 PCA : 13
# acc :  0.6111111044883728
# 걸린 시간 :  1.98 초
# ===========================
# 결과 PCA : 13
# acc :  0.6111111044883728
# 걸린 시간 :  9.52 초
# ===========================
