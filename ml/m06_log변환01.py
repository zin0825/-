import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)
print(data)
print(data.shape)   # (1000,)
print(np.min(data), np.max(data))   # 0.0021589800723233136 14.748205974501298


# log_data = np.log(data)
log_data = np.log1p(data)
# 로그에는 1이 없는데 데이터에는 1이 있을 수 있음
# 그래서 log가 아닌 log1p를 사용해서 1(p)을 더 해줌


# 원본 데이터 히스토그램 그리자
plt.subplot(1, 2, 1)   # 1바이 1짜리
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title("Original")
# plt.show()


# 로그변환 데이터 히스토르갬 그리자
plt.subplot(1, 2, 2)   # 1바이 2짜리
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformd')
plt.show()   # 데이터가 옮겨갔음










