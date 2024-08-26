# 89이상
# 다중분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from  tensorflow.keras. callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = ".\\_data\\keggle\\otto-group-product-classification-challenge\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv.shape)   # (61878, 94)
# print(test_csv.shape)   # (144368, 93)
# print(sample_csv.shape)   # (144368, 9)

# print(train_csv.isnull().sum())
# print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['target'] = encoder.fit_transform(train_csv['target'])
print(train_csv.shape)

x = train_csv.drop(['target'], axis=1)
print(x.shape)   # [(61878, 93)

y = train_csv['target']
print(y.shape)   # (61878,)

y = pd.get_dummies(y)  
print(y)   # [61878 rows x 9 columns]


print(x.shape, y.shape)   # (160, 13) (18, 13)

exit()


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


n = []
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=93))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(130, activation='relu'))
    model.add(Dense(130, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    #3. 컴파일,  훈련
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

    path = './_save/m05_13/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'm05_13_date_', str(i+1),'_', date, '_epo_', filename])  


    ######################### cmp 세이브 파일명 만들기 끗 ###########################

    mcp = ModelCheckpoint( # mcp는 ModelCheckpoint
    monitor='val_loss',
    mode='auto',
    verbose=0,
    save_best_olny=True, 
    filepath = filepath,
    )

    hist = model.fit(x_train, y_train, epochs=700, batch_size=2232, 
                    validation_batch_size=0.3,
                    callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")
    print('===========================')


sample_csv[['Class_1',	'Class_2',	'Class_3',	'Class_4',	'Class_5',	
            'Class_6',	'Class_7',	'Class_8',	'Class_9']] = y_submit

sample_csv.to_csv(path + "sampleSubmission_0823_1132.csv")


# acc score :  0.7334356819650937
# 걸린 시간 :  4.17 초
# 로스 :  [0.5940346717834473, 0.7771493196487427]

# acc score :  0.7805429864253394
# 걸린 시간 :  370.3 초
# 로스 :  [2.9506893157958984, 0.7813510298728943]

# acc score :  0.7618778280542986
# 걸린 시간 :  103.16 초
# 로스 :  [0.762648344039917, 0.7828054428100586]

# 스켈링
# acc score :  0.7462831286360698
# 걸린 시간 :  7.19 초
# 로스 :  [0.5738803744316101, 0.7899159789085388]

# acc score :  0.7429702650290886
# 걸린 시간 :  48.89 초
# 로스 :  [0.6805111169815063, 0.7722204327583313]