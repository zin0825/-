# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

# 맹그러!!!!
# 다중분류인줄 알았더니 이진분류였다!!!
# 다중분류 다시 찾겠노라!!!

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
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape)   # (200000, 201)
# print(test_csv.shape)   # (200000, 200)
# print(sample_csv.shape)   # (200000, 1)

# print(train_csv.columns)

x = train_csv.drop(['target'], axis=1)
# print(x)   # [200000 rows x 200 columns]

y = train_csv['target']
# print(y)

print(x.shape, y.shape)   # (200000, 200) (200000,)

# exit()

pca = PCA(n_components=200)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 111
# print(np.argmax(cumsum >= 0.99) +1)  # 144
# print(np.argmax(cumsum >= 0.999) +1) # 174
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=2038,
                                                    stratify=y)

# print(x)
# print(y)
# print(x.shape, y.shape)   # (200000, 200) (200000,)


n = [111, 144, 174, 1]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=n[i]))
model.add(Dense(400))
model.add(Dense(340))
model.add(Dense(330))
model.add(Dense(250))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
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

path = './_save/m05_12/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'm05_12_date_', str(i+1),'_', date, '_epo_', filename])  


######################### cmp 세이브 파일명 만들기 끗 ###########################

mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
monitor='val_loss',
mode='auto',
verbose=0,
save_best_olny=True, 
filepath = filepath,
)


hist = model.fit(x_train1, y_train, epochs=50, batch_size=1000, 
                 verbose=0, validation_batch_size=0.3,
                 callbacks=[es, mcp])
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test1, y_test, verbose=0)

print('결과 PCA :', n[i] )
print('acc : ', loss[1])
print('걸린 시간 : ', round(end - start, 2), "초")
print('===========================')



# acc score :  0.87495
# 걸린 시간 :  104.54 초
# 로스 :  [1.0752360820770264, 0.8749499917030334]

# batch_size=1000
# acc score :  0.897375
# 걸린 시간 :  117.65 초
# 로스 :  [0.3148635923862457, 0.8973749876022339]

# batch_size=5000
# acc score :  0.9049
# 걸린 시간 :  58.69 초
# 로스 :  [0.25515782833099365, 0.9049000144004822]

# acc score :  0.91195
# 걸린 시간 :  60.1 초
# 로스 :  [0.24282044172286987, 0.9119499921798706]

# acc score :  0.91185
# 걸린 시간 :  58.62 초
# 로스 :  [0.24183888733386993, 0.9118499755859375]

# 스켈링
# acc score :  0.912525
# 걸린 시간 :  120.47 초
# 로스 :  [0.2372458279132843, 0.9125249981880188]

