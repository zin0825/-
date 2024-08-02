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
from sklearn.metrics import r2_score, accuracy_score
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




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


x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, x_test.shape)   # (160, 200, 200, 1) (120, 200, 200, 1)

x_train = x_train.reshape(160, 200, 200, 1)
x_test = x_test.reshape(120, 200, 200, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(200, 200, 1), activation='relu',
                 strides=1, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu', 
                 strides=1, padding='same'))
model.add(Conv2D(16, (2,2), strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True)



######################### cmp 세이브 파일명 만들기 끗 ###########################

import datetime   # 날짜
date = datetime.datetime.now()   
print(date)  
print(type(date))  
date = date.strftime("%m%d_%H%M")   
print(date)   
print(type(date))



path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   


filepath = "".join([path, 'k41_02',date, '_' , filename]) 


######################### cmp 세이브 파일명 만들기 끗 ###########################


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True, # 가장 좋은 놈을 저장
    filepath = filepath)    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# 파일네임, 패스 더하면 요놈

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=316,
          verbose=1, 
          validation_split=0.3,
          callbacks=[es, mcp])
end = time.time()



#4. 평가, 예측      <- dropout 적용 X
loss = model.evaluate(x_test, y_test, verbose=1)

print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('r2_score : ', r2)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)


# acc :  0.99
# r2_score :  0.9580706541385943
# 걸린 시간 :  10.02 초
# 로스 :  [0.04775671288371086, 0.9916666746139526]

# acc :  0.99
# r2_score :  0.9614636287050715
# 걸린 시간 :  12.7 초
# 로스 :  [0.04310063645243645, 0.9916666746139526]