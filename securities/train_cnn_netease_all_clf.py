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

stock_codes = [600082, 600169, 600036, 600201, 300104]

"""
#get all codes
today = datetime.now().strftime('%Y-%m-%d')
df = ts.get_day_all(today)

if df is None or df.empty:
    print('failed to get codes')
    exit(-1)

stock_codes = df['code'].tolist()
"""

snapshot_dir = './snapshots_pick/train_cnn_netease_all_clf'
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

pick_dir = './snapshots_pick/pick_cnn_netease_all_clf'
if not os.path.exists(pick_dir):
    os.makedirs(pick_dir)

log_file = './snapshots_pick/train.log'
log = open(log_file, 'w')

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


def train_model_by_code(code):
    hist_data_path = os.path.join(path, str(code) + '_D.csv')
    
    if not os.path.isfile(hist_data_path):
        print('hist data not exists:%s' % hist_data_path)
        exit(-1)
    
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
    max_score = 0
    max_index = -1
    max_cp_path = None
    score_threshold = 0.54
    for i in range(4):
        cp_path = os.path.join(snapshot_dir, str(code) + '_D_' + str(i) + '.hdf5')
        model_cp = ModelCheckpoint(cp_path, save_best_only=True, monitor='val_acc', mode='max')
        cb_lists = [model_cp]
    
        model.fit(X_train,
                  y_train,
                  epochs=70,
                  batch_size=64,
                  verbose=1,
                  shuffle=True,
                  validation_split=0.05,
                  callbacks=cb_lists)
    
        model.load_weights(filepath=cp_path)
        score = model.evaluate(X_test, y_test, batch_size=50)
    
        if score[1] > max_score: 
            max_score = score[1]

            if score[1] > score_threshold:
                max_index = i
                max_cp_path = cp_path
    
    if max_index < 0:
        print('No valid model:%s' % (code))
        log.write('No valid model %s:%s\n' % (code, max_score))
        return
    
    print('best snapshot score:%.2f%%' % (max_score * 100) )
    log.write('%s best score:%.2f%%\n' % (code, max_score * 100))

    best_cp_path = os.path.join(pick_dir, '%s_D_%s.hdf5' % (code, max_score))
    os.rename(max_cp_path, best_cp_path)
    
    
    model.load_weights(filepath=best_cp_path)
    pred_y_test = model.predict_classes(X_test)
    #print(pred_y_test)
    pred_y_test = pred_y_test.astype(np.int64)
    #print(pred_y_test)
    y_test = np.argmax(y_test, axis=1)
    #print(y_test)
    print(classification_report(y_test, pred_y_test))

for code in stock_codes:
    train_model_by_code(code)
    log.flush()

log.close()
