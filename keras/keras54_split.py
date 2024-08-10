# import numpy as np
# a = np.array(range(1, 11))
# size = 5

# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)   # append 추가하다
#     return np.array(aaa)
    
    
# bbb = split_x(a, size)
# print(bbb)
# # [[ 1  2  3  4  5]
# #  [ 2  3  4  5  6]
# #  [ 3  4  5  6  7]
# #  [ 4  5  6  7  8]
# #  [ 5  6  7  8  9]
# #  [ 6  7  8  9 10]]
# print(bbb.shape)   # (6, 5)


# x = bbb[:, :-1]
# y = bbb[:, -1]
# print(x, y)
# # [[1 2 3 4]
# #  [2 3 4 5]
# #  [3 4 5 6]
# #  [4 5 6 7]
# #  [5 6 7 8]
# #  [6 7 8 9]] [ 5  6  7  8  9 10]
# print(x.shape, y.shape)   # (6, 4) (6,)



# time steps = 과거 몇개의 데이터를 볼 것인가를 나타내며, 
# 네트워크에 사용할 수 있는 데이터의 양을 결정
# feater = x의 차원을 의미, x의 변수 갯수


import numpy as np
a = np.array(range(1, 11))
size = 5


def split_x(dataset, size):   # split_x 함수를 정의하고 매개변수를 넣겠다 / split 함수를 정의 한다
    aaa = []   # aaa라는 빈 리스트
    for i in range(len(dataset) - size + 1):   # len = 길이 / dataset의 갯수 만큼 for문을 돌려서 dataset의 길이에서 size를 뺀 값에 1을 더한 만큼 반복
        subset = dataset[i : (i + size)]   # 반복 할 때 사용되는 데이터의 길이(수)는 여기서 결정
                                           # subset은 dataset의 i부터 i+size 까지 넣겠다. 1회전 할 때 i는 0이므로 0 + 5 = 5 마지막 숫자는 5
        aaa.append(subset)   # append 추가하다 / aaa에 subset을 추가하겠다 
    return np.array(aaa)   # 넘파이 배열로 변환
    
    
bbb = split_x(a, size)   # 스플릿_x라는 함수에 a와 사이즈를 입력 / split를 표출해준다
print(bbb)
print(bbb.shape)   # (6, 5)


x = bbb[:, :-1]   # x의 값을 가져와서 마지막 줄(행)을 제외한다
y = bbb[:, -1]   # y의 값을 가져와서 
print(x, y)
print(x.shape, y.shape)   # (6, 4) (6,)
