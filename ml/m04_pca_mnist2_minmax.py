
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   #  (2중)분류방식

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)   # (60000,) (10000,)
# DNN에서 60000, 28, 28을 곱했었음
# mnist를 리쉐잎했을 때  28 * 28 = 784
# 4개를 반환해야하는데 나머지는 필요 없을 때 _로 받음
# 전체 데이터의 PCA를 해볼거야 70000만개를. 왜 7만개? 그냥 해볼거야
# 연산을 하는게 아니라 그냥 붙일거야

### mnist = 이미지라서 스레일링 할 필요 없음

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)   #  (70000, 28, 28)


# 스케일링/ 추가
x = x/255.
print(np.min(x), np.max(x))   # 0.0 1.0


# exit()

########## [실습] #########
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일때 몇개?

# 힌트 np.argmax

# x = x.reshape(70000, 28*28)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# 쉐이프를 명확히 안다면 좋지만 모르거나 변형을 할 수 있을때 위 방식으로 한다

print(x.shape)   # (70000, 784)

# exit()

pca = PCA(n_components=784)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_


cumsum = np.cumsum(evr)
print(cumsum)

# 쌤 코드
# print(np.argmax(evr_cumsum >= 1.0))   # 712로 나옴. 그래서 밑에것 처럼 +1 해줌
print(np.argmax(cumsum >= 0.95) + 1)   # 154
print(np.argmax(cumsum >= 0.99) +1)   # 331
print(np.argmax(cumsum >= 0.999) + 1)   # 486
print(np.argmax(cumsum >= 1.0) + 1)   # 713


# exit()


# 챗gpt 코드
# n_components_95 = np.argmax(cumsum >= 0.95) + 1
# n_components_99 = np.argmax(cumsum >= 0.99) + 1
# n_components_999 = np.argmax(cumsum >= 0.999) + 1
# n_components_1 = np.argmax(cumsum >= 1.0) + 1

# print(f"n_components >= 0.95: {n_components_95}")   # n_components >= 0.95: 154
# print(f"n_components >= 0.99: {n_components_99}")   # n_components >= 0.99: 331
# print(f"n_components >= 0.999: {n_components_999}") # n_components >= 0.999: 486
# print(f"n_components >= 1.0: {n_components_1}")     # n_components >= 1.0: 713


# print('e.95 이상 :', np.min(np.where(cumsum>-0.95)))
# print('0.99 이상 :', np.min(np.where(cumsum>-0.99)))
# print('0.999 이상 :', np.min(np.where(cumsum>-0.999)))
# print('1.0 일 때 :', np.argmax(cumsum))


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()












