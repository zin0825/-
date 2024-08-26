# keras18_overfit1_boston


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA



#1. 데이터 
dataset = load_boston()
print(dataset)
# print(dataset.DESCR)   # describe 확인 
# print(dataset.feature_names)


x = dataset.data   # x 데이터 분리   # 스켈링 할 것, x만 (비율만) 건들고 y는 건들면 안됨
y = dataset.target   # y 데이터 분리, sklearn 문법

# print(x.shape)   # (506, 13)
# print(y.shape)   # (506, )


pca = PCA(n_components=13)  
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_ 

cumsum = np.cumsum(evr) 

# print(np.argmax(cumsum >= 0.95) +1)  # 2
# print(np.argmax(cumsum >= 0.99) +1)  # 3
# print(np.argmax(cumsum >= 0.999) +1) # 6
# print(np.argmax(cumsum >= 1.0) +1)   # 1

# exit()

# x_train과 x_test 하기전에 분리...를 여기보다

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True, 
                                                    random_state=333)



n = [2, 3, 6, 1]
results = []

########## for 문 #############

for i in range(0, len(n), 1):
    pca = PCA(n_components = n[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)


    #2. 모델 구성
    model = Sequential()
    # model.add(Dense(10, input_dim=13))   # 특성은 많으면 좋음, 한계가 있음, 인풋딤에 다차원 행렬이 들어가면 안됨 
    model.add(Dense(32, input_dim=n[i]))   # 이미지 input_shape=(8,8,1) ,하나 있는건 벡터이기 때문
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(1))


    #3. 컴파일, 훈련
    model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])

    start = time.time()

    es = EarlyStopping(monitor='val_loss', mode='min',
                    patience=5, verbose=1,
                    restore_best_weights=True)

    model.fit(x_train1, y_train, epochs=100, batch_size=32,
            verbose=1, 
            validation_split=0.2,
            callbacks=[es]
            )

    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    print('결과 PCA :', n[i] )
    print('acc : ', loss[1])
    print('걸린 시간 : ', round(end - start, 2), "초")



# pca
# 결과 PCA : 2
# acc :  0.0
# 걸린 시간 :  2.47 초

# 결과 PCA : 3
# acc :  0.0
# 걸린 시간 :  3.08 초

# 결과 PCA : 6
# acc :  0.0
# 걸린 시간 :  2.27 초

# 결과 PCA : 1
# acc :  0.0
# 걸린 시간 :  2.67 초