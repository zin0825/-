from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
#  Layer (type)                Output Shape              Param #   # param연산
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 4)                 16

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 1)                 4
# =================================================================
# Total params: 41
# Trainable params: 41
# Non-trainable params: 0   # 나중에 쓸 일 있음, 훈련을 하지 않고 남이 쓴걸 낼름

# dense_1(4_) x dense(3) + 1 =16
# dense_2(3) x dense_1(5) + 1 =15
# dense_3(1) x dense_2(4) + 1 =4
# 바이어스 때문에 / 항상 뒤에 바이어스 있음
# h1 = w1 X x1 + w2 X x2 + b1
# h2 = w3 X x1 + w4 X x2 + b2
# h2 = w5 X x1 + w6 X x2 + b3


