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
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=5,   # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,   # 축소 또는 화대, 1.2배
    shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)


start1 = time.time()


np_path = 'C:\\프로그램\\ai5\\_data\\_save_npy\\'

x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,  # test 안 할 때 주의!!
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=44)



print(x_train)
print(x_train.shape)   # 
print(y_train)
print(y_train.shape)   #


augment_size =  10000

print(x_train.shape[0]) 

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)
print(x_train[0].shape)


x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]

print(x_augmented.shape) 

print(x_train.shape, x_test.shape)


x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) 


end1 = time.time()

print('걸린 시간1 : ', round(end1 - start1, 2), "초")



#2. 모델 구성
model = Sequential()
model.add(Conv2D(110, (2,2), input_shape=(100, 100, 3), activation='relu',
                 strides=1, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=110, kernel_size=(2,2), activation='relu', 
                 strides=1, padding='same'))
model.add(Conv2D(110, (2,2), strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(110, activation='relu',))
model.add(Dense(100, activation='relu',))
model.add(Dense(80, activation='relu',))
model.add(Dense(20, activation='relu',))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)



######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   # 현재 시간
print(date)   # 2024-07-26 16:50:13.613311
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   # 시간을 strf으로 바꾸겠다
print(date)   # "%m%d" 0726  "%m%d_%H%M" 0726_1654
print(type(date))



path = './_save/keras49/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
filepath = "".join([path, 'k49_07_',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것

######################### cmp 세이브 파일명 만들기 끗 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
)   # 파일네임, 패스 더하면 요놈



hist = model.fit(x_train, y_train, epochs=50, batch_size=200,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])

end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1)



y_pred = model.predict(x_test)

y_pred = np.round(y_pred)
# print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('acc : ', accuracy)
print('r2_score : ', r2)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)




# acc :  0.996031746031746
# 로스 :  [0.019349033012986183, 0.9960317611694336]


# acc :  1.0
# r2_score :  1.0
# 걸린 시간 :  89.5 초
# 로스 :  [0.000641791382804513, 1.0]

# acc :  1.0
# r2_score :  1.0
# 걸린 시간 :  86.22 초
# 로스 :  [0.00030748904100619256, 1.0]