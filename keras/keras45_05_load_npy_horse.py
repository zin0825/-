

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sklearn as sk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



train_datagen = ImageDataGenerator(
    rescale=1./255,)


test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문

path_train = './_data/image/horse_human/'   # 상위 폴더를 불러야함   
path_test = './_data/image/horse_human/' 


start1 = time.time()

xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(100,100),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=20000,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='rgb',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(100,100),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=20000,   
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   


print(xy_train[0][0].shape)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]



print(x_train)
print(x_train.shape)   # 
print(y_train)
print(y_train.shape)   #
# print(x_test)
# print(x_test.shape)   # 
# print(y_test)
# print(y_test.shape)   # 


np_path = 'c:/프로그램/ai5/_data/_save_npy/'
# np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train)
# np.save(np_path + 'kares45_01_y_train.npy', arr=xy_train[0][1])   # 통데이터로 저장해야 함
# np.save(np_path + 'kares45_01_x_test.npy', arr=xy_test[0][0])  
# np.save(np_path + 'kares45_01_y_test.npy', arr=xy_test[0][1])  

x_train = np.load(np_path + 'keras45_02_x_train.npy')
y_train = np.load(np_path + 'kares45_02_y_train.npy')
x_test = np.load(np_path + 'kares45_02_x_test.npy')
y_test = np.load(np_path + 'kares45_02_y_test.npy')




end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1, 2), "초")


print("======================== 2. MCP 출력 ====================")

path2 = 'C:\\프로그램\\ai5\\_save\\keras41\\'
model = load_model(path2 + 'k41_04_0805_2124_0043-0.0164.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)


print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('걸린 시간 : ', round(end1 - start1, 2), "초")
print('로스 : ', loss)



# 걸린 시간1 :  12.1 초
# acc :  0.994157740993184
# 걸린 시간 :  12.1 초
# 로스 :  [0.020223306491971016, 0.9941577315330505]
