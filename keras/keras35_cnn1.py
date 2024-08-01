# 수치화 = 이미지를 조각조각 쪼갬
# 조각을 내서 합치고x 5 해서 특정 특성을 냈음
# 반복해서 사진을 가져왔는데 눈밑에 다크서클, 뿔테안경 그 부분이 수치가 높음

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D   # 컨블루션 2D와 같음. 이미지기 때문에 2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))   # (4, 4, 10)
# 커널(2,2) 이미지를 자르겠다 필터10만큼 복사 / (5,5,1) 가로, 세로, 컬러
model.add(Conv2D(5, (2,2)))   # (3, 3, 5)

model.summary()
# Model : "Sequntiol"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0








