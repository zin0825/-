import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 아워너 80

print(x.shape, y.shape)   # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)   # (13, 3, 1)


#2. 모델 구성
model = Sequential()
model.add(LSTM(60, input_shape=(3,1)))
model.add(Dense(60, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))



model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()


# es = EarlyStopping(monitor='loss', mode='min',
#                    patience=10,
#                    verbose=1,
#                    restore_best_weights=True)

# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))
# date = date.strftime("%m%d_%H%M")
# print(date)
# print(type(date))

# path = './_save/keras52_02_summary'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #'1000-0.7777.hdf5' (파일 이름. 텍스트)
# # {epoch:04d}-{val_loss:.4f} fit에서 빼와서 쓴것. 쭉 써도 되는데 가독성이 떨어지면 안좋음
# # 로스는 소수점 이하면 많아지기 때문에 크게 잡은것
# filepath = "".join([path, 'k42_02_summary',date, '_' , filename])    # 문자열을 만드는데 아무것도 없는 공문자를 만들고
# # 생성 예: ""./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5"   그냥 텍스트 파일. 문자를 생성한것


# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True, # 가장 좋은 놈을 저장
#     filepath = filepath    # 좋은놈이 계속 갱신하면서 저장하기 때문에 1개만 있음
# )   # 파일네임, 패스 더하면 요놈


# hist = model.fit(x, y, epochs=1500, batch_size=32,
#           verbose=1, 
#           validation_split=0.1,
#           callbacks=[es, mcp])


np_path = 'c:/프로그램/ai5/_save/keras52_02_summary'
model.save("c:/프로그램/ai5/_save/keras52_02_summary/k52_02_summary04.h5")


model.fit(x, y, epochs=1500)


end = time.time()


#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = np.array([50,60,70]).reshape(1, 3, 1)
y_pred = model.predict(x_pred)

print('[50,60,70]의 결과 :', y_pred)




# model.add(LSTM(60, input_shape=(3,1)))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(56, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(1))
# epochs=1500)
# loss :  0.020263465121388435
# [50,60,70]의 결과 : [[78.79388]]

# loss :  0.0003102876653429121
# [50,60,70]의 결과 : [[79.2954]]

# loss :  5.209277333051432e-06
# [50,60,70]의 결과 : [[79.16645]]
# keras52_02_summaryk42_02_summary0807_1701_0025-1896.7566


# loss :  4.3760410335380584e-05
# [50,60,70]의 결과 : [[79.331665]]
# k52_02_summary02


# @@@@@@
# loss :  9.343335113953799e-05
# [50,60,70]의 결과 : [[79.33196]]  
# k52_02_summary03
# @@@@@@


# epochs=1200
# loss :  1.0061169632535893e-05
# [50,60,70]의 결과 : [[79.04367]]
# k52_02_summary04