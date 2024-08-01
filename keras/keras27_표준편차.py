import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
# data = np.array([[1,2,3],
#                  [4,5,6],
#                  [7,8,9],
#                  [10,11,12],
#                  [13,14,15]])

# 첫열 평균 7, 이열 8, 삼열 9

# data = np.array([[1,2,3,1],
#                  [4,5,6,2],
#                  [7,8,9,13],
#                  [10,11,12,14],
#                  [13,14,15,15]])

# 7, 8, 9, 9 한쪽으로 치우쳐짐

# data = np.array([[1,2,3,1],
#                  [4,5,6,2],
#                  [7,8,9,3],
#                  [10,11,12,114],
#                  [13,14,15,115]])


data = np.array([[4,5,3,9],
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]])



#1. 평균
means = np.mean(data, axis=0)   # 헷갈리면 아무거나 넣어
print('평균 : ', means)   # 평균 :  [ 7.  8.  9. 47.]

#2. 모집단 분산 (n빵)   # ddof=0 디폴트
population_variances = np.var(data, axis=0, )
print("모집단 분산 : ", population_variances)   
# 모집단 분산 :  [  18.   18.   18. 3038.]

# #3. 표본 분산 (n-1빵)
variances = np.var(data, axis=0, ddof=1)
print("표본 분산 : ", variances)   
# 표본 분산 :  [  22.5   22.5   22.5 3797.5]

#4. 표본 표준편차
std = np.std(data, axis=0, ddof=1)
print("표본 표준편차 : ", std)   
# 표본 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5. StandardScaler
scaler = StandardScaler()

scaler_data = scaler.fit_transform(data)

print("StandardScaler : \n", scaler_data)  # ***
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]
#                                          평균 47







