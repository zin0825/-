
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

path_np = 'C:\\프로그램\\ai5\\_data\\image\\me\\'
np.save(path_np + 'me_arr.npy', arr=img)