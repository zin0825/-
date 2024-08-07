
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img   #이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   #땡겨온거 수치화
import matplotlib.pyplot as plt
import numpy as np



path = 'C:\\프로그램\\ai5\\_data\\image\\me\\me.jpg'

img = load_img(path, target_size=(100, 100),)
print(img)

print(type(img))

# plt.imshow(img)
# plt.show()


arr = img_to_array(img)
print(arr)
print(arr.shape)   # (146, 180, 3) -> (200, 200, 3)
print(type(arr))   # <class 'numpy.ndarray'>


# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)   # (1, 100, 100, 3)

# me 폴더에 위에 데이터를 npy로 저장할것   / 넘파이로 저장

# path_np = 'C:\\프로그램\\ai5\\_data\\image\\me\\'
# np.save(path_np + 'me_arr.npy', arr=img)


############## 요기부터 증폭 ####################

datagen = ImageDataGenerator(
    rescale=1./255,   # 1 나누기 255, 0에서 스켈일링
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,   # 수직 뒤집기
    width_shift_range=0.2,   # 평행이동
    # height_shift_range=0.1,   # 평행이동 수직 (위 아래로)
    rotation_range=15,   # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,   # 축소 또는 화대, 1.2배
    # shear_range=0.7,   # 좌표 하나를 고정시키고 다른 몇개의 좌료를 이동시키는 변환. (개의 입을 고정 시키고 턱을 벌린다)
    fill_mode='nearest',   # 너의 빈자리 비슷한거로 채워줄게
)


it = datagen.flow(img,   # 디렉토리부터 가져오겠다 / flow 수치화된 데이터를 바꾼다
                  batch_size=1)   

# print(it)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001B5272E3FD0>   # 이터레이터 형태
# 선택적으로 한번 뽑아서 끝 x n@

# print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))

for i in range(5):
    batch = it.next()
    print(batch.shape)   # TypeError: Invalid shape (1, 100, 100, 3) for image data / 얘를 리쉐이프 해준다
    batch = batch.reshape(100,100,3)
    
    
    ax[i].imshow(batch)
    ax[i].axis('off')
        
plt.show()






