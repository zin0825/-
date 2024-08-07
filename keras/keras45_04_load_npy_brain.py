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



#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,   # 수직 뒤집기
    # width_shift_range=0.1,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    # rotation_range=5,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    # fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문

path_train = './_data/image/brain/train/'   
path_test = './_data/image/brain/test/' 


start1 = time.time()


xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(200,200),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=160,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='grayscale',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(200,200),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=160,   
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   



print(xy_train[0][0].shape)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]



# print(x_train)
# print(x_train.shape)   # 
# print(y_train)
# print(y_train.shape)   #
# print(x_test)
# print(x_test.shape)   # 
# print(y_test)
# print(y_test.shape)   # 



np_path = 'c:/프로그램/ai5/_data/_save_npy/'
x_train = np.load(np_path + 'keras45_01_01_x_train.npy')
y_train = np.load(np_path + 'kares45_01_01_y_train.npy')
x_test = np.load(np_path + 'kares45_01_01_x_test.npy')
y_test = np.load(np_path + 'kares45_01_01_y_test.npy')





end1 = time.time()

print('걸린 시간 : ', round(end1 - start1, 2), "초")



print("======================== 2. MCP 출력 ====================")

path2 = 'C:\\프로그램\\ai5\\_save\\keras41\\'
model = load_model(path2 + 'k41_02_0805_2123_0100-0.0570.hdf5')  

loss = model.evaluate(x_test, y_test, verbose=1, batch_size=12)

y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)


print('걸린 시간1 : ', round(end1 - start1,2), "초")

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('걸린 시간 : ', round(end1 - start1, 2), "초")
print('로스 : ', loss)



# 걸린 시간1 :  0.3 초
# acc :  0.9916666666666667
# 걸린 시간 :  0.3 초
# 로스 :  [0.04917959123849869, 0.9916666746139526]
