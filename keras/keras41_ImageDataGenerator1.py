import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

test_datagen = ImageDataGenerator(
    rescale=1./255,)   # 테스트 데이터는 절대 변환하지 않고 수치화만 한다. 평가해야하기 때문, 동일한 규격과 동일한 조건으로만 하기 때문

path_train = './_data/image/brain/train/'   # 라벨이 분리 된 상위 폴더까지 잡음
path_test = './_data/image/brain/test/'   # 라벨이 분리 된 상위 폴더까지 잡음

xy_train = train_datagen.flow_from_directory(   # 수치화
    path_train,   # 이 폴더 안에 있는 걸 다 수치화
    target_size=(200,200),   # 사진은 사이즈가 제각각이라서 사이즈를 모두 동일하게, 큰건 축소, 작은건 증폭
    batch_size=10,   # 데이터가 80개 있는데 10개씩 묶어서 훈련  80 x 10, 200, 200, 1 y
    class_mode='binary',   # binary 이진법
    color_mode='grayscale',   # 흑백
    shuffle=True,   # 섞겠다
    )   # Found 160 images belonging to 2 classes.   브레인 트레인 80, ad 80 = 160

xy_test = test_datagen.flow_from_directory(  
    path_test, 
    target_size=(200,200),   # 10, 200, 200, 1 200개의 데이터가 10개 있음 -> (트레인) 16개 생김
    batch_size=10,   
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=False   # 해도 상관은 없지만 셔플을 할 필요가 없다. 원래 (위치) 그대로 써야하기 때문
    )   # Found 120 images belonging to 2 classes.
# 첫번째 폴더 전부 0으로 때림

print(xy_train)   # <keras.preprocessing.image.DirectoryIterator object at 0x00000232598F2CA0>

# print(xy_train.next())   # array([1., 0., 1., 1., 1., 1., 1., 1., 0., 1.], dtype=float32))
# # 이 데이터의 첫번째 데이터를 보여줘   10 x 10, 200, 200, 1
# # x와 y 데이터가 모여있는 이터레이터 형태
# # 폴더채로 되어있지 않고 셔플되어 있음
# # 보스톤에서 분리 함. 분리되기 전이 이터레이터
# #  x = dataset.data   # skleran 문법 데이터 분리
# # y = dataset.target   
# print(xy_train.next())   # array([1., 0., 0., 1., 0., 1., 0., 0., 1., 0.], dtype=float32))
# # 반복되기에 for문 사용 for : in 이터레이터

print(xy_train[0])   # 셔플이 있어서 조금씩 다름
# array([1., 1., 0., 0., 1., 1., 1., 0., 0., 1.], dtype=float32))
# 첫번째 x라 0, 0

print(xy_train[0][0])

print(xy_train[0][1])   # 0, 1 y

# print(xy_train[0].shape)   # 튜플상태 0번째 x, 첫번째 y
# AttributeError: 'tuple' object has no attribute 'shape'

print(xy_train[0][0].shape)   # (10, 200, 200, 1) [0]16개,배치,0부터15,x[0]2개, 각각의 01,01,01

# print(xy_train[16])   # 15까지라 없음
# ValueError: Asked to retrieve element 16, but the Sequence has length 16
# retrieve 앞으로 자주 나올 예정

# print(xy_train[15][2])
# IndexError: tuple index out of range

print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>

print(type(xy_train[0]))   # <class 'tuple'>

print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>
# 배치를 160으로 잡았다면 160,200,200,1










