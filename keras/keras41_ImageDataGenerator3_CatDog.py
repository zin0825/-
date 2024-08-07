# 캣독 맹그러봐!!!

#1. 에서 시간체크
# 배치를 160으로 잡고
# x, y를 추출해서 모델을 맹그러봐
# acc 0.99이상
'''
batsh_size=160
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
'''

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


start = time.time()

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

path_train = 'C:\\프로그램\\ai5\\_data\\image\\cat_and_dog\\Train'   
path_test = 'C:\\프로그램\\ai5\\_data\\image\\cat_and_dog\\Test' 


xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(80,80),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=20000,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='rgb',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(80,80),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=20000,   
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


print(xy_train[0][0].shape)


np_path = 'c:/프로그램/ai5/_data/_save_npy/'
np.save(np_path + 'keras41_03_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras41_03_01_y_train.npy', arr=xy_train[0][1])   # 통데이터로 저장해야 함
np.save(np_path + 'keras41_03_01_x_test.npy', arr=xy_test[0][0])  
np.save(np_path + 'keras41_03_01_y_test.npy', arr=xy_test[0][1])  



# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1],
#                                                     train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=66)



# end = time.time()

# print('걸린 시간 : ', round(end - start,2), "초")

# # 걸린 시간 :  41.42 초

# print(x_train.shape, x_test.shape)   # (15997, 100, 100, 3) (4000, 100, 100, 3)
# print(y_train.shape, y_test.shape)   # (15997,) (4000,)




# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(18, (2,2), input_shape=(150, 150, 3), activation='relu',
#                  strides=1, padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=18, kernel_size=(2,2), activation='relu', 
#                  strides=1, padding='same'))
# model.add(Conv2D(18, (2,2), strides=1, padding='same'))
# model.add(Flatten())

# model.add(Dense(10, activation='relu',))
# model.add(Dense(10, activation='relu',))
# model.add(Dense(8, activation='relu',))
# model.add(Dense(2, activation='relu',))
# model.add(Dense(1, activation='sigmoid'))


# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# start = time.time()

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10,
#                    verbose=1,
#                    restore_best_weights=True)


# ######################### cmp 세이브 파일명 만들기 끗 ###########################

# import datetime   # 날짜
# date = datetime.datetime.now()   # 현재 시간
# print(date)   # 2024-07-26 16:50:13.613311
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
# print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
# print(type(date))



# path = './_save/keras41/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# filepath = "".join([path, 'k41_03_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

# ######################### cmp 세이브 파일명 만들기 끗 ###########################


# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True, # 가장 좋은 놈을 저장
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )   # 파일네임, 패스 더하면 요놈


# hist = model.fit(x_train, y_train, epochs=60, batch_size=22,
#           verbose=1, 
#           validation_split=0.3,
#           callbacks=[es, mcp])
# end = time.time()



# #4. 평가, 예측      <- dropout 적용 X
# loss = model.evaluate(x_test, y_test, verbose=1)



# y_pred = model.predict(x_test)

# y_pred = np.round(y_pred)
# # print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print('acc : ', accuracy)
# print('r2_score : ', r2)
# print('걸린 시간 : ', round(end - start,2), "초")
# print('로스 : ', loss)



# acc :  0.4925
# r2_score :  -1.0304568962374048
# 걸린 시간 :  23.13 초
# 로스 :  [0.6931678056716919, 0.4925000071525574]

# acc :  0.49725
# r2_score :  -1.0110606571453729
# 걸린 시간 :  56.08 초
# 로스 :  [0.6931719183921814, 0.4972499907016754]

# acc :  0.48675
# r2_score :  -1.0544427064381159
# 걸린 시간 :  50.13 초
# 로스 :  [0.6932852268218994, 0.4867500066757202]