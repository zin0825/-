from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier   #  (2중)분류방식
from sklearn.ensemble import RandomTreesEmbedding   # 회귀방식
from sklearn.decomposition import PCA   # PCA 백테와 비슷
# ecomposition 분해
# 디스전트리 앙승블이랑 비슷함

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
# print(x.shape, y.shape)   # (150, 4) (150,)

# pca하면 데이터가 변환됨 -> 피쳐가 들어듬 ->차원을 축소한다면 01010이 아니라 0.몇으로 바뀜
# pca 스케일링이랑 같이 쓰면 좋음
# 파라미터 튜닝과 전처리의 개념이라 다 이해하고 할 줄 알아야함


scaler = StandardScaler()   # 스켈러 역시 y는 안함
x = scaler.fit_transform(x)


pca = PCA(n_components=4)   # 4개의 컬럼이 3개로 자름 / 0, 5로 하면 안됨
x = pca.fit_transform(x)
# 비지도학습이지만 전처리 개념으로 봐야함

# scaler = StandardScaler()   # 스켈러 역시 y는 안함
# x = scaler.fit_transform(x)   # pca의 개념적인거라 통으로 해도 됨 (트레인 테스트 별도 x)
# pca, scaler의 순서는 상관없지만 통상적으로 pca먼저 하는게 성능 좋음

# print(x)
# print(x.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=10, 
                                                    shuffle=True,
                                                    stratify=y)   # y의 클래스(라벨)에 맞춰서 트레인 스플릿 비율에 맞춰서 한다

#2. 모델
model = RandomForestClassifier(random_state=10)   # 트리계열 찾아보기


# 머신러닝은 컴파일 할 필요 없음. 모델만 하면 됨
#3. 훈련
model.fit(x_train, y_train)   # 디폴트 100으로 잡혀있음


#4. 평가, 예측
# 모델.이벨류 가 없음
# 이그레서는 R2스코어로 평가

results = model.score(x_test, y_test)
print(x.shape)
print('model.score : ', results)


# model.score :  0.9

# random_state=333
# pca = PCA(n_components=3)   
# x = pca.fit_transform(x)   안한거
# model.score :  0.9333333333333333


# pca = PCA(n_components=3)   
# x = pca.fit_transform(x)
# (150, 3)
# model.score :  0.9


# pca = PCA(n_components=2)   
# x = pca.fit_transform(x)
# (150, 2)
# model.score :  0.9333333333333333


# pca = PCA(n_components=1)   
# x = pca.fit_transform(x)
# (150, 1)
# model.score :  0.9333333333333333


# 스켈러 먼저함
# (150, 3)
# model.score :  0.9333333333333333


# pca = PCA(n_components=4)
# train_size=0.9,
# random_state=10, 
# (150, 4)
# model.score :  1.0


# pca = PCA(n_components=3)
# # train_size=0.9,
# # random_state=10, 
# (150, 3)
# model.score :  1.0

# pca = PCA(n_components=2)
# train_size=0.9,
# random_state=10, 
# (150, 2)
# model.score :  1.0

# pca = PCA(n_components=1) 
# train_size=0.9,
# random_state=10, 
# (150, 1)
# model.score :  1.0


# pca = PCA(n_components=1) 
# train_size=0.8,
# random_state=10, 
# (150, 1)
# model.score :  1.0

# pca = PCA(n_components=4)
# train_size=0.9,
# random_state=10, 
# (150, 4)
# model.score :  0.9666666666666667

# pca = PCA(n_components=3)
# # train_size=0.8,
# # random_state=10, 
# (150, 3)
# model.score :  1.0

# pca = PCA(n_components=2)
# train_size=0.8,
# random_state=10, 
# (150, 2)
# model.score :  1.0

# pca = PCA(n_components=1) 
# train_size=0.8,
# random_state=10, 
# (150, 1)
# model.score :  1.0
