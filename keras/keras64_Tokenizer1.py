
# 토근화
# 음절
# 어절
# 형태소
# 말을 토큰나이징 (조각조각) 해서 수치화 시키겠다


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
# text = '나는 지금 주차료가 엄청 엄청 많이 많이 많이 나올까봐 매우 매우 불안하다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# 많이 나오는 단어 순서, 먼저 나오는 단어 순서 대로 인덱스
# {'많이': 1, '엄청': 2, '매우': 3, '나는': 4, '지금': 5, '주차료가': 6, '나올까봐': 7, '불안하다': 8}

print(token.word_counts)   # 몇 번 반복 횟수
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])   # 리스트. 나중에 두 개 이상 쓸 수 있다
print(x)   # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]   (1, 9)
# print(x.shape)   # 리스트는 shape가 없다. 정 알고 싶을 땐 앞에 len을 붙여준다
# 수치화 했어도 이대로 사용할 수 없다. 원핫인코딩을 해야 한다.
# 진짜 x 진짜 = 한다고 해서 매우가 되지 않기 때문에 원핫인코딩을 해야 한다.
# 원핫인코딩을 하면 전체 단어의 개수만큼의 열이 생긴다. 각 단어의 인덱스에 해당하는 열에 1을 넣어준다.


# x = len(x)[[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]


# 케라스 투 카테고리, 판다스, 사이킷런 원핫인코딩


# x = to_categorical(x)
# print(x)  
# print(x.shape)
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 14, 10)


# x = pd.DataFrame(x)
# x = pd.get_dummies(x)
# print(x)
# print(x.shape) 
#    0   1   2   3   4   5   6   7   8   9   10  11  12  13
# 0   4   5   2   2   3   3   6   7   8   1   1   1   1   9
# (1, 14)
 

# # x = pd.DataFrame(x)
# x = pd.get_dummies(x)
# print(x)
# print(x.shape)   # (14, 9, 10)


# ohe = OneHotEncoder(sparse=True)
# x = ohe.fit_transform(x)
# print(x)   
# print(x.shape)   
#  (0, 0)        1.0
#   (0, 1)        1.0
#   (0, 2)        1.0
#   (0, 3)        1.0
#   (0, 4)        1.0
#   (0, 5)        1.0
#   (0, 6)        1.0
#   (0, 7)        1.0
#   (0, 8)        1.0
#   (0, 9)        1.0
#   (0, 10)       1.0
#   (0, 11)       1.0
#   (0, 12)       1.0
#   (0, 13)       1.0
# (1, 14)



#### 위에 데이터를 (14,9)로 바꿔라. 0빼!

# 케라스

x = to_categorical(x, num_classes=10)   # 쌤 코드. 9개 넣으면 에러 
print(x)  
print(x.shape)


# x = np.array(x).reshape(-1, 1)
# x = to_categorical(x)
# x = x[:, 1:]
# print(x)  
# print(x.shape)
# [[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (14, 9)




# 판다스

# x = np.array(x).reshape(-1,)   # 쌤 코드
# # x = pd.DataFrame(x)
# x = pd.get_dummies(x)
# print(x)
# print(x.shape)   # (14, 9, 10)


# x_flatten = [item for sublist in x for item in sublist]
# # flatten은 numpy에서 제공하는 다차원 배열 공간을 1차원으로 평탄화해주는 함수
# print(x_flatten)   # pandas Series로 변환
# x_series = pd.Series(x_flatten)   # get_dummies로 원핫 인코딩
# x = pd.get_dummies(x_series)   # series는 판다스 1차원 배열로 변환
# print(x)
# print(x.shape)
# 0   0  0  0  1  0  0  0  0  0
# 1   0  0  0  0  1  0  0  0  0
# 3   0  1  0  0  0  0  0  0  0
# 4   0  0  1  0  0  0  0  0  0
# 5   0  0  1  0  0  0  0  0  0
# 6   0  0  0  0  0  1  0  0  0
# 7   0  0  0  0  0  0  1  0  0
# 8   0  0  0  0  0  0  0  1  0
# 9   1  0  0  0  0  0  0  0  0
# 10  1  0  0  0  0  0  0  0  0
# 11  1  0  0  0  0  0  0  0  0
# 12  1  0  0  0  0  0  0  0  0
# 13  0  0  0  0  0  0  0  0  1
# (14, 9)




# 사이킷런
# x = np.array(x).reshape(-1, 1)   # 쌤 코드와 동일
# ohe = OneHotEncoder(sparse=True)
# x = ohe.fit_transform(x)
# print(x)   
# print(x.shape)   
#   (0, 3)        1.0
#   (1, 4)        1.0
#   (2, 1)        1.0
#   (3, 1)        1.0
#   (4, 2)        1.0
#   (5, 2)        1.0
#   (6, 5)        1.0
#   (7, 6)        1.0
#   (8, 7)        1.0
#   (9, 0)        1.0
#   (10, 0)       1.0
#   (11, 0)       1.0
#   (12, 0)       1.0
#   (13, 8)       1.0
# (14, 9)


