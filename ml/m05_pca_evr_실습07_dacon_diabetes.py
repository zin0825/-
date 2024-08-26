# https://dacon.io/competitions/open/236068/data
# 풀어라!!!


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터

path = "./_data/dacon/biabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)   # [652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)   # [116 rows x 8 columns]

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(sample_submission_csv)   # [116 rows x 1 columns]

# print(train_csv.shape, test_csv.shape, sample_submission_csv.shape)
# # (652, 9) (116, 8) (116, 1)

# print(train_csv.columns) # 9개 -1

# print(train_csv.info())

# print(train_csv.isna().sum())

# # train_csv = train_csv.dropna()
# # print(train_csv.isna().sum())

# print(train_csv)   # [652 rows x 9 columns]

test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())


x = train_csv.drop(['Outcome'], axis=1)

# print(x)   # [652 rows x 8 columns]

y = train_csv['Outcome']
# print(y)

# print(x.shape)   # (652, 8)
# print(y.shape)   # (652,)

# exit()

pca = PCA(n_components=8)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 2
# print(np.argmax(cumsum >= 0.99) +1)  # 5
# print(np.argmax(cumsum >= 0.999) +1) # 6
# print(np.argmax(cumsum >= 1.0) +1)   # 8

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=7575)


n = [2, 5, 6, 8]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=n[i]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(148, activation='relu'))
    model.add(Dense(168, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    start = time.time()

    es = EarlyStopping(monitor= 'val_loss', 
                       mode='auto',
                       patience=20, 
                       restore_best_weights=True)

    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    import datetime
    date = datetime.datetime.now()
    # print(date)    
    # print(type(date))  
    date = date.strftime("%m%d_%H%M")
    # print(date)     
    # print(type(date))  

    path = './_save/m05_07/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_07_date_', str(i+1),'_', date, '_epo_', filename])  


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



y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_predict.shape)   # (116, 1)
y_submit = np.round(y_submit)
sample_submission_csv['Outcome'] = y_submit


sample_submission_csv.to_csv(path + "sample_sbmission_0822_1522.csv")


# 로스 :  0.27064642310142517
# ACC :  0.667

# 로스 :  0.2617517113685608
# ACC :  0.667
# 걸린시간 :  4.47 초

# 로스 :  [0.21972215175628662, 0.6818181872367859]

# 로스 :  [0.19491037726402283, 0.6818181872367859]

# 로스 :  [0.24453414976596832, 0.6818181872367859]

# 로스 :  [0.23192520439624786, 0.7121211886405945]

# 로스 :  [0.1500083953142166, 0.7575757503509521]

# 로스 :  [0.2740616202354431, 0.7121211886405945]

# 로스 :  [0.2821880877017975, 0.7121211886405945]

# 로스 :  [4.80870246887207, 0.6515151262283325]

# 로스 :  [2.4672627449035645, 0.5757575631141663]

# pca
# 결과 PCA : 2
# acc :  0.7121211886405945
# 걸린 시간 :  2.48 초
# ===========================
# 결과 PCA : 5
# acc :  0.7121211886405945
# 걸린 시간 :  2.18 초
# ===========================
# 결과 PCA : 6
# acc :  0.6969696879386902
# 걸린 시간 :  2.22 초
# ===========================
# 결과 PCA : 8
# acc :  0.7272727489471436
# 걸린 시간 :  2.0 초
# ===========================