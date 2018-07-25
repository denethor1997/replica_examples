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

data_dir = './data/netease/hist_ma/'
code = 600082
#code = 600169
#code = 600815
#code = 600036
#code = 300104
#code = 600201
#code = '002608'

stock_codes = [600082, 600169, 600036, 600201]

snapshot_dir = './snapshots_pick/pick_cnn_netease_all_clf'
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


def test_model_by_code(code):
    hist_data_path = os.path.join(data_dir, str(code) + '_D.csv')
    
    if not os.path.isfile(hist_data_path):
        print('hist data not exists:%s' % hist_data_path)
        exit(-1)
    
    X, y, dates = get_data_label_dates(hist_data_path)
    #X, y, dates = get_data_label_dates(hist_data_path_fq, reverse=False)
    
    dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    
    X_test = X[-1:]
    y_test = y[-1:]
    print(X_test.shape)
    
    total = 32
    pad_h_l = (total - X_test.shape[1])//2
    pad_h_r = total - X_test.shape[1] - pad_h_l
    pad_w_t = (total - X_test.shape[2])//2
    pad_w_b = total - X_test.shape[2] - pad_w_t
    #padding
    X_test = np.pad(X_test, ((0,0), (pad_h_l,pad_h_r), (pad_w_t,pad_w_b)),'constant')
    print(X_test.shape)
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    print(X_test.shape)
    
    """
    #repeat 3 channels
    X_train = np.repeat(X_train, 3, axis=3)
    X_test = np.repeat(X_test, 3, axis=3)
    print(X_train.shape)
    print(X_train[0])
    """
    
    model = clf_cnn_prelu((X_test.shape[1], X_test.shape[2], X_test.shape[3]))
   
    best_cp_path = None
    for path in os.listdir(snapshot_dir):
        if path.startswith(str(code) + '_D'):
            best_cp_path = os.path.join(snapshot_dir, path)
            break
    
    model.load_weights(filepath=best_cp_path)
    pred_y_test = model.predict(X_test)

    return pred_y_test.ravel()

code_scores = []
for code in stock_codes:
    pred_y = test_model_by_code(code)
    print('%s:%.2f%%' % (code, pred_y[0] * 100))
    code_scores.append(pred_y[0])

sorted_code_scores = np.argsort(np.array(code_scores))
#print(sorted_code_scores)

for i in range(3):
    max_index = sorted_code_scores[-1 - i]
    #print(max_index)
    print('picked %s:%.2f%%' % (stock_codes[max_index], code_scores[max_index] * 100))

