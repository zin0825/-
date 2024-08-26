# https://www.kaggle.com/competitions/playground-series-s4e1

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)      # (165034, 13)
# print(test_csv.shape)       # (110023, 12)
# print(mission_csv.shape)    # (110023, 1)

# print(train_csv.columns)
# # Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
# #        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
# #        'EstimatedSalary', 'Exited'],

# print(train_csv.isnull().sum())     # 결측치가 없다
# print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis=1)
print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
print(y.shape)                      # (165034,)

# print(x.shape, y.shape)   # (165034, 10)(165034,)


# exit()

pca = PCA(n_components=10)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 8
# print(np.argmax(cumsum >= 0.99) +1)  # 10
# print(np.argmax(cumsum >= 0.999) +1) # 10
# print(np.argmax(cumsum) +1)   # 10

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9,
                                                    shuffle=True,
                                                    random_state=1866)

n = [8, 10, 10, 10]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(32, input_dim=n[i]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    start = time.time()

    es = EarlyStopping(monitor='val_loss', mode = 'min', 
                       patience=10, restore_best_weights=True,)

    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    import datetime
    date = datetime.datetime.now()
    # print(date)    
    # print(type(date))  
    date = date.strftime("%m%d_%H%M")
    # print(date)     
    # print(type(date))  

    path = './_save/m05_08/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_08_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train1, y_train, epochs=100, batch_size=80, 
                    verbose=0, validation_split=0.2, 
                    callbacks=[es, mcp])

    end = time.time()


    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')


y_submit = model.predict(test_csv)
print(y_submit.shape)       # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_0822_1534.csv")

# 결과 PCA : 8
# acc :  0.8457343578338623
# 걸린 시간 :  240.77 초
# ===========================
# 결과 PCA : 10
# acc :  0.864820659160614
# 걸린 시간 :  118.76 초
# ===========================
# 결과 PCA : 10
# acc :  0.8643965125083923
# 걸린 시간 :  92.55 초
# ===========================
# 결과 PCA : 10
# acc :  0.8648812174797058
# 걸린 시간 :  129.3 초
# ===========================
# (110023, 1)