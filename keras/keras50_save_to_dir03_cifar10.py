
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


train_datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.2,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=15,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

augment_size = 40000


print(x_train.shape[0])   # 50000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)   # [42254 31185  9620 ... 29307  5350 29357]

print(x_train[0].shape)   # (32, 32, 3)


x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)   # (40000, 32, 32, 3) (40000, 1)


# x_augmented = x_augmented.reshape(x_augmented.shape[0],
#                                   x_augmented.shape[1],
#                                   x_augmented.shape[2],
#                                   x_augmented.shape[3], 1)
# print(x_augmented.shape)   # (40000, 32, 32, 3, 1)   
# # 컬러 3이 있어서 따로 reshape 안해도 됨


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                 save_to_dir='c:/프로그램/ai5/_data/_save_img/03_cifar10').next()[0]

print(x_augmented.shape)   # (40000, 32, 32, 3)




"""
print(x_train.shape, x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)   # (90000, 32, 32, 3) (90000, 1)


### 원핫 y 1-1 케라스 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델

input1 = Input(shape=(32,32,3))   # = input_dim = (3,)))
convd1 = Conv2D(128, (3,3), activation='relu', name='ys1')(input1)   # 앞에 레이어의 이름이 명시   # 함수형은 순차형보다 자유로움
maxpool1 = MaxPooling2D((2,2))(convd1)
convd2 = Conv2D(64, (2,2), activation='relu', strides=1, 
                padding='valid',name='ys2')(maxpool1)
convd3 = Conv2D(64, (2,2), activation='relu', name='ys3')(convd2) 
drop1 = Dropout(0.3)(convd3)
convd4 = Conv2D(32, (2,2),activation='relu', name='ys4')(drop1)
flatten = Flatten()(convd4)
dense1 = Dense(16, name='ys5')(flatten)
drop2 = Dropout(0.3)(dense1)
dense2 = Dense(10, name='ys6')(drop2)
output1 = Dense(10)(dense2)
model = Model(inputs=input1, outputs=output1)   # 모델의 명시를 마지막에 함
model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)


start = time.time()
hist = model.fit(x_train, y_train, epochs=15, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('로스 : ', loss)
print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print(y_pred)

accureacy = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('acc_score : ', acc)
print('걸린 시간 : ', round(end - start,2), "초")
print('로스 : ', loss)

"""

# acc_score :  0.394
# 걸린 시간 :  18.47 초
# 로스 :  [1.7561897039413452, 0.39399999380111694]

# acc_score :  0.6755
# 걸린 시간 :  67.0 초
# 로스 :  [0.9785659909248352, 0.6754999756813049]


# stride
# acc_score :  0.7289
# 걸린 시간 :  35.45 초
# 로스 :  [0.7947065830230713, 0.7289000153541565]


# MaxPooling2D / valid
# acc_score :  0.5455
# 걸린 시간 :  40.32 초
# 로스 :  [1.7251884937286377, 0.5454999804496765]

# acc_score :  0.6705
# 걸린 시간 :  43.68 초
# 로스 :  [0.9671040177345276, 0.6704999804496765]

# same
# acc_score :  0.7224
# 걸린 시간 :  43.09 초
# 로스 :  [0.8071021437644958, 0.7224000096321106]


# acc_score :  0.0313
# 걸린 시간 :  55.63 초
# 로스 :  [2.2554056644439697, 0.031300000846385956]


# augmente
# acc_score :  0.0259
# 걸린 시간 :  60.67 초
# 로스 :  [2.1710619926452637, 0.02590000070631504]

# acc_score :  0.1717
# 걸린 시간 :  52.0 초
# 로스 :  [2.259598970413208, 0.17170000076293945]

