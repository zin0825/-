from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 파이썬 버전 달라서 없으면 찾을 것. 분명 어딘가에 있음
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()   
# 판다스로 데이터 프레임을 자유자재로 쓰는게 중요. 데이터셋 변경하는거 확실히 배워둘 것
# x_train 사이즈 = (60000, 28, 28)


train_datagen = ImageDataGenerator(
    # rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.2,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=15,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)

augment_size = 100   # 아그먼트. 증가시키다. 변수명도 이름 잘 생각하면서 만들것

print(x_train.shape)   # (60000, 28, 28)
print(x_train[0].shape)   # (28, 28) -> 잘못된거


# plt.imshow(x_train[0], cmap='gray')
# plt.show()   # 코랩은 쇼 안쳐도 나오지만 나머지는 다 해야함


aaa =np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)
# (60000, 28, 28)
# (28, 28)
# (100, 28, 28, 1)


xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
             batch_size=augment_size,   # 데이터가 100개 대괄호가 두개 있을 때 앞이 x 뒤가 y
             shuffle=False,).next()   # next()를 쓰면 무조건 한번 만 돌아감
# np.tile(x_train[0].reshape(28*28), augment_size) = np.tile. 다차원 배열이라 reshape로 1차원 배열로 만들어줌

# 리스트 대괄호 안에 거 바꿀수 있다. 튜플 중괄호 안에거 바꿀 수 없다.
# np.zeros(augmnet_size) = y값을 넣어줘야 하는데 없어서 0으로 넣어줌





print(xy_data)   # 내가 생각한것과 다르게 나올 수도 있으니 꼭 찍어볼 것
# print(x_data.shape)   # AttributeError: 'tuple' object has no attribute 'shape'

print(type(xy_data))   # <class 'tuple'> 추가 하그라!!!

print(len(xy_data))   # 2 튜플 x와 y가 존재한다.
print(xy_data[0].shape)   # 넘파이라서 shape가 존재.   (100, 28, 28, 1)
print(xy_data[1].shape)   # (100,)


plt.figure(figsize=(7,7))   # 알아서 사이즈 조절
for i in range(49):
    plt.subplot(7, 7, i+1)   # 7,7, 에 0번째를 넣어라..
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()




