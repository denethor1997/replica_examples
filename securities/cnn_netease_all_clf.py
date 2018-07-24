# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import datetime as dt
import os
import time
from datetime import date

import numpy as np
from sklearn import preprocessing
import tushare as ts
import pandas as pd

from keras.callbacks import ModelCheckpoint

from utils.load_data import *
from models.rmse import *
from models.clf_cnn import clf_cnn, clf_cnn_prelu
#from models.reg_mobilenet import reg_mobilenet


from sklearn.metrics import classification_report

path = './data/netease/hist_ma/'
code = 600082
#code = 600169
#code = 600815
#code = 600036
#code = 300104
#code = 600201
#code = '002608'

snapshot_dir = './snapshots/cnn_netease_all_clf'
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

def get_data_label_dates(path, reverse=True):
    df = pd.read_csv(path)
    features = []
    targets = []
    dates = []

    volumes = []
    for index, row in df.iterrows():
        vols = []
        vols.append(row['volume'])
        vols.append(row['v_ma5'])
        vols.append(row['v_ma10'])
        vols.append(row['v_ma20'])

        volumes.append(vols)

    volumes = preprocessing.scale(volumes)

    for index, row in df.iterrows():
        day_prices = []
        day_prices.append(row['open'])
        day_prices.append(row['close'])
        day_prices.append(row['high'])
        day_prices.append(row['low'])
        day_prices.append(row['ma5'])
        day_prices.append(row['ma10'])
        day_prices.append(row['ma20'])

        if 'turnover' in row:
            day_prices.append(row['turnover'])

         
        row_date = row['date']
        tokens = row_date.split('-')
        real_date = date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        day_prices.append(real_date.month/12.0)
        day_prices.append(real_date.day/31.0)
        day_prices.append(real_date.weekday()/7.0)
        
       
        features.append(day_prices + volumes[index].tolist())
        #features.append(day_prices)

        #targets.append(row['close'])
        targets.append(row['ma5'])
        
        dates.append(row['date'])

    if reverse:
        features = features[::-1]
        targets = targets[::-1]
        dates = dates[::-1]

    slide_window = 15
    dayn = 3 #start from 0
    data = []
    label = []
    label_dates = []
    for i in range(len(dates) - slide_window - 1 - dayn):
        data.append(features[i:i + slide_window])
        label.append([1,0] if targets[i + slide_window + dayn] - targets[i + slide_window - 1] > 0 else [0,1])
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

total = 32
pad_h_l = (total - X_train.shape[1])//2
pad_h_r = total - X_train.shape[1] - pad_h_l
pad_w_t = (total - X_train.shape[2])//2
pad_w_b = total - X_train.shape[2] - pad_w_t
#padding
X_train = np.pad(X_train, ((0,0), (pad_h_l,pad_h_r), (pad_w_t,pad_w_b)),'constant')
X_test = np.pad(X_test, ((0,0), (pad_h_l,pad_h_r), (pad_w_t,pad_w_b)),'constant')
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

model = clf_cnn_prelu((X_train.shape[1], X_train.shape[2], X_train.shape[3]))

best_cp_path = os.path.join(snapshot_dir, str(code) + '_D_@.hdf5')
model_cp = ModelCheckpoint(best_cp_path, save_best_only=True, monitor='val_acc', mode='max')
cb_lists = [model_cp]

model.fit(X_train,
          y_train,
          epochs=70,
          batch_size=64,
          verbose=1,
          shuffle=True,
          validation_split=0.05,
          callbacks=cb_lists)

model.load_weights(filepath=best_cp_path)
score = model.evaluate(X_test, y_test, batch_size=50)
print "****************************************"
print 'code =', code
print "****************************************"

print ""
print "test : ", model.metrics_names[0], score[0], model.metrics_names[1], score[1]
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

os.rename(best_cp_path, best_cp_path.replace('@', str(score[1])))

pred_y_test = model.predict_classes(X_test)
#print(pred_y_test)
pred_y_test = pred_y_test.astype(np.int64)
#print(pred_y_test)
y_test = np.argmax(y_test, axis=1)
#print(y_test)
print(classification_report(y_test, pred_y_test))

"""
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
"""
