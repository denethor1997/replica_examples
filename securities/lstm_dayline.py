# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import datetime as dt
from utils.load_data import *
#from data_preprocess.Extract_Features import Extract_Features
from sklearn import preprocessing
from models.rmse import *
from models.reg_lstm import reg_lstm

path = './data/stock_data/'
#code = 600841
#code = 600082
#code = 600083
#code = 601718
#code = 600196
#code = 600519
code = 600815
#code = 600036

dates = []
oneDayLine = []
thirtyDayLine = []
month_dates = []

if os.path.isfile(path + str(code) + '.csv'):
    pass
else:
    download_from_tushare(code)

oneDayLine, dates = load_data_from_tushare(path+str(code)+'.csv')
thirtyDayLine, month_dates = load_data_from_tushare(path+str(code)+'_month.csv')
dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

#ef = Extract_Features()
#X, y = To_DL_datatype(code, scale=True)
X, y = To_DL_datatype_ma5(code, scale=True)
#X, y = To_DL_datatype_p_change(code, scale=False)

#X = preprocessing.scale(X)
#y = preprocessing.scale(y)
print(X[0])
print(X[1])
print(X[2])
print(y[0])
print(y[1])

X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, y, 0.8)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = reg_lstm(10)
model.fit(X_train,
          Y_train,
          nb_epoch=300,
          batch_size=50,
          verbose=1,
          validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=50)
print score
predicted_Y_test = model.predict(X_test)

print(X_test[0])
print(X_test[1])
print(X_test[2])
print(X_test[3])
print(Y_test[0])
print(predicted_Y_test[0])

print(X_test[-2])
print(Y_test[-2])
print(predicted_Y_test[-2])
tmp = X_test[-2]
tmp = tmp.reshape(1,1,10)
print(model.predict(tmp))

print(X_test[-1])
print(Y_test[-1])
print(predicted_Y_test[-1])
tmp = X_test[-1]
tmp = tmp.reshape(1,1,10)
print(model.predict(tmp))

# means of val
my_rmse = rmse(predicted_Y_test, Y_test)
print "rmse = ", my_rmse

plt.figure(1)
#plt.ylim(-1.5,3)
plt.plot(dates[len(dates)-len(Y_test):len(dates)], Y_test, color='g')
plt.plot(dates[len(dates)-len(predicted_Y_test):len(dates)], predicted_Y_test, color='r')
plt.show()

predicted_Y_train = model.predict(X_train)
plt.figure(1)
plt.plot(dates[0:len(Y_train)], Y_train, color='g')
plt.plot(dates[0:len(predicted_Y_train)], predicted_Y_train, color='r')
plt.show()
