# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import datetime as dt
import os
import time

import numpy as np
from sklearn import preprocessing
import tushare as ts
import pandas as pd

from utils.load_data import *
from models.rmse import *
from models.reg_cnn import reg_cnn
from models.reg_mobilenet import reg_mobilenet

path = './data/netease/hist_ma'
code = 600082
#code = 600169
#code = 600815
#code = 600036
#code = 300104


def get_data_label_dates(path, reverse=True):
    df = pd.read_csv(path)
    prices = []
    close_prices = []
    dates = []
    for index, row in df.iterrows():
        day_prices = []
        day_prices.append(row['open'])
        day_prices.append(row['close'])
        day_prices.append(row['high'])
        day_prices.append(row['low'])
        
        prices.append(day_prices)

        #close_prices.append(row['close'])
        close_prices.append(row['ma5'])
        
        dates.append(row['date'])
    
    if reverse:
        prices = prices[::-1]
        close_prices = close_prices[::-1]
        dates = dates[::-1]

    slide_window = 15
    dayn = 1 #start from 0
    data = []
    label = []
    label_dates = []
    for i in range(len(dates) - slide_window - 1 - dayn):
        data.append(prices[i:i + slide_window])
        label.append(close_prices[i + slide_window + dayn])
        label_dates.append(dates[i + slide_window + dayn])

    return np.array(data), np.array(label), np.array(label_dates)

hist_data_path = os.path.join(path, str(code) + '_D.csv')

if os.path.isfile(hist_data_path):
    pass
else:
    download_from_tushare(code)

#hist_data_path_fq = os.path.join(path, str(code) + '_fq.csv')

X, y, dates = get_data_label_dates(hist_data_path)
#X, y, dates = get_data_label_dates(hist_data_path_fq, reverse=False)

dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

X_train, X_test, y_train, y_test = create_Xt_Yt(X, y, 0.95)
print(X_train.shape)

#padding
X_train = np.pad(X_train, ((0,0), (11,11), (14,14)),'constant')
X_test = np.pad(X_test, ((0,0), (11,11), (14,14)),'constant')
print(X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
print(X_train.shape)

"""
#repeat 3 channels
X_train = np.repeat(X_train, 3, axis=3)
X_test = np.repeat(X_test, 3, axis=3)
print(X_train.shape)
print(X_train[0])
"""

model = reg_cnn((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
model.fit(X_train,
          y_train,
          epochs=800,
          batch_size=64,
          verbose=1,
          #shuffle=True,
          validation_split=0.1)

score = model.evaluate(X_test, y_test, batch_size=50)
print score
pred_y_test = model.predict(X_test)

# means of val
my_rmse = rmse(pred_y_test, y_test)
print "rmse = ", my_rmse

plt.figure(1)
#plt.ylim(-1.5,3)
plt.plot(dates[len(dates)-len(y_test):len(dates)], y_test, color='g')
plt.plot(dates[len(dates)-len(pred_y_test):len(dates)], pred_y_test, color='r')
plt.show()

pred_y_train = model.predict(X_train)
plt.figure(1)
plt.plot(dates[0:len(y_train)], y_train, color='g')
plt.plot(dates[0:len(pred_y_train)], pred_y_train, color='r')
plt.show()
