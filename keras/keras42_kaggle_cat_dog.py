# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview


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

path_trainc = 'C:/프로그램/ai5/_data/keggle/dogs-vs-cats-redux-kernels-edition/'   
path_traind = 'C:/프로그램/ai5/_data/keggle/dogs-vs-cats-redux-kernels-edition/'
path_test = 'C:/프로그램/ai5/_data/keggle/dogs-vs-cats-redux-kernels-edition/'


xy_trainc = train_datagen.flow_from_directory(   # 수치화
    path_trainc,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(200,200),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=80,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='categorical',   ## 다중분류 - 원핫 인코딩도 나와서 따로 할 필요 xx
    # class_mode='sparse',   ## 다중분류 
    # class_mode='binary'   # 이진분류
    # class_mode=None   # y값 없다!!!
    # color_mode='grayscale',   
    color_mode='rgb',
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160


xy_traind = train_datagen.flow_from_directory(   # 수치화
    path_traind,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(200,200),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=80,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='categorical',   ## 다중분류 - 원핫 인코딩도 나와서 따로 할 필요 xx
    # class_mode='sparse',   ## 다중분류 
    # class_mode='binary'   # 이진분류
    # class_mode=None   # y값 없다!!!
    # color_mode='grayscale',   
    color_mode='rgb',
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160



xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(200,200),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=80,   
    class_mode='categorical',
    color_mode='rgb', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]   # 라벨이 몇개 인지 확인
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


# print(xy_train[0][0].shape)
# color_mode='grayscale',
# (30, 100, 100, 1)
# color_mode='rgb',
# (30, 100, 100, 3)


x_train, x_test, y_train, y_test = train_test_split(xy_trainc[0][0], xy_trainc[0][1],
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=44)



end = time.time()

print('걸린 시간 : ', round(end - start,2), "초")

# # 걸린 시간 :  3.99 초

print(x_train.shape, y_test.shape)   # (64, 200, 200, 3) (16, 4)
print(y_train.shape, y_test.shape)   # (64, 4) (16, 4)




#2. 모델 구성
model = Sequential()
model.add(Conv2D(80, (2,2), input_shape=(200, 200, 3), activation='relu',
                 strides=1, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=80, kernel_size=(2,2), activation='relu', 
                 strides=1, padding='same'))
model.add(Conv2D(80, (2,2), strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(60, activation='relu',))
model.add(Dense(50, activation='relu',))
model.add(Dense(20, activation='relu',))
model.add(Dense(20, activation='relu',))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)





start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es])
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






