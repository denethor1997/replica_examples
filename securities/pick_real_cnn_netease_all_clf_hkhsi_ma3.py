# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import datetime as dt
import os
import time
from datetime import date, datetime
import gc

import numpy as np
from sklearn import preprocessing
import tushare as ts
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from utils.load_data import *
from models.rmse import *
from models.clf_cnn import *
#from models.reg_mobilenet import reg_mobilenet


from sklearn.metrics import classification_report

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)


data_dir = './data/netease/hist_ma/'
code = 600082
#code = 600169
#code = 600815
#code = 600036
#code = 300104
#code = 600201
#code = '002608'

#stock_codes = [600082, 600169, 600036, 600201, 600400, 600448, 600536, 600339, 600103, 600166]

df = ts.get_day_all()

if df is None or df.empty:
    print('failed to get codes')
    exit(-1)

stock_codes = df['code'].tolist()

pick_index = -1

snapshot_dir = './snapshots_pick/pick_cnn_netease_all_clf_hkhsi_ma3'
if not os.path.exists(snapshot_dir):
    print('snapshot dir not exists:%s' % snapshot_dir)
    exit(-1)

ts = str(datetime.now()).replace(' ', '@').replace(':', '_')
log_path = os.path.join('snapshots_pick', 'real_hkhsi_ma3_%s.log'%ts)
log = open(log_path, 'w')

def get_data_dates_hkhsi():
    csvpath = './data/hkHSI_D.csv'
    df = pd.read_csv(csvpath)
    data = []
    dates = []

    for index, row in df.iterrows():
        features = []
        features.append(row['open'])
        features.append(row['close'])
        features.append(row['high'])
        features.append(row['low'])
        features.append(row['volume'])
        data.append(features)
    
        dates.append(row['date'])

    data = preprocessing.scale(data)

    return np.array(data), np.array(dates) 

hkhsi_data, hkhsi_dates = get_data_dates_hkhsi()
print(hkhsi_data)


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
        day_prices.append(row['ma3'])

        if 'turnover' in row:
            day_prices.append(row['turnover'])

         
        row_date = row['date']
        tokens = row_date.split('-')
        real_date = date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        day_prices.append(real_date.month/12.0)
        day_prices.append(real_date.day/31.0)
        day_prices.append(real_date.weekday()/7.0)
        
        #add hkhsi index
        for hkindex, hkdate in enumerate(hkhsi_dates):
            if hkdate <= row_date:
                day_prices += hkhsi_data[hkindex].tolist()
                break
       
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
    for i in range(len(dates) - slide_window + 1):
        data.append(features[i:i + slide_window])
        label_dates.append(dates[i + slide_window - 1])

    return np.array(data), np.array(label), np.array(label_dates)

def get_model_by_code(code):
    best_cp_path = None
    max_acc = 0.0
    for path in os.listdir(snapshot_dir):
        if path.startswith(str(code) + '_D'):
            cp_path = os.path.join(snapshot_dir, path)
            tokens = cp_path.split('@')
            if len(tokens) < 4:
                print('invalid path:%s'%cp_path)
                continue

            acc = float(tokens[1])
            
            precs = tokens[2].split('_')
            up_prec = float(precs[0])
            down_prec = float(precs[1])
            
            recall = tokens[3].split('_')
            down_recall = float(recall[1])

             
            if acc < 0.7 or up_prec < 0.7 or down_prec < 0.6 or down_recall < 0.5:
                continue
            
            if max_acc > acc:
                continue
            
            max_acc = acc
            best_cp_path = cp_path

            #break; #???
    
    return best_cp_path

def test_model_by_code(code):
    hist_data_path = os.path.join(data_dir, str(code) + '_D.csv')
    
    if not os.path.isfile(hist_data_path):
        print('hist data not exists:%s' % hist_data_path)
        return [0], '', ''
 
    
    best_cp_path = get_model_by_code(code)
    """
    for path in os.listdir(snapshot_dir):
        if path.startswith(str(code) + '_D'):
            best_cp_path = os.path.join(snapshot_dir, path)
            break
    """

    if best_cp_path is None:
        print('no model for %s' % code)
        return [0], '', ''

   
    X, y, dates = get_data_label_dates(hist_data_path)
    #X, y, dates = get_data_label_dates(hist_data_path_fq, reverse=False)
    
    dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
   
    X_test = X[pick_index]
    X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
    date_test = dates[pick_index]
    print(X_test.shape)
    print(date_test)
    #log.write('%s\n'%date_test)

    if (X_test.shape[0] <= 0):
        print('no data for %s' % code)
        return [0], '', date_test

    close1 = X_test[0][-1][1]
    close2 = X_test[0][-2][1]
    close4 = X_test[0][-4][1]
    close5 = X_test[0][-5][1]
    if close1 + close2 > (close4 + close5)*1.015:
        print('ignore data for %s(%s,%s,%s,%s)' % (code,close1,close2,close4,close5))
        return [0], 'ignore data', date_test

    
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
    
    model = clf_cnn_prelu_categorical((X_test.shape[1], X_test.shape[2], X_test.shape[3]))
   
    model.load_weights(filepath=best_cp_path)
    pred_y_test = model.predict(X_test)

    del model
    K.clear_session()
    gc.collect()

    return pred_y_test.ravel(), best_cp_path, date_test

code_scores = []
code_results = []
for code in stock_codes:
    pred_y, best_cp_path, date_test = test_model_by_code(code)
    print('%s:%.2f%%' % (code, pred_y[0] * 100))
    log.write('%s(%s):%.2f%%, date:%s\n' % (code, best_cp_path, pred_y[0] * 100, date_test))
    log.flush()
    code_scores.append(pred_y[0])

    ret = [pred_y[0],best_cp_path,date_test]
    code_results.append(ret)

sorted_code_scores = np.argsort(np.array(code_scores))
#print(sorted_code_scores)

high_total = 0
high_correct = 0

low_total = 0
low_correct = 0

for i in range(len(code_scores)):
    max_index = sorted_code_scores[-1 - i]

    if code_scores[max_index] <= 0:
        break

    rets = code_results[max_index]
    score = rets[0]
    best_cp_path = rets[1]
    date_test = rets[2]
    #print(max_index)
    print('picked %s:%.2f%%' % (stock_codes[max_index], code_scores[max_index] * 100))
    log.write('picked %s:%.2f%%,%s,%s\n' % (stock_codes[max_index], code_scores[max_index] * 100, date_test,best_cp_path))
    log.flush()

log.close()
