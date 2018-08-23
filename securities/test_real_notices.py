import datetime as dt

import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.netease import *
from datetime import datetime

data_dir = './data/netease/hist_ma/'

stock_codes = [600082, 600169, 600036, '000999']

today = datetime.now().strftime('%Y-%m-%d')

pick_index = -2

df = ts.get_day_all()

if df is None or df.empty:
    print('failed to get codes')
    exit(-1)

stock_codes = df['code'].tolist()
print('total codes:%s' % len(stock_codes))

timestr = datetime.now().strftime('%Y-%m-%d')
log_path = os.path.join('snapshots_pick', 'test_notices_%s.log'%timestr)
log = open(log_path, 'w')

def get_notices_by_code(code, curdate):
    try:
        notices = ts.get_notices(code=str(code), date=curdate)
        return notices
    except Exception as e:
        print(e)
        return None

def get_label_dates(code, reverse=True):
    path = os.path.join(data_dir, str(code) + '_D.csv')

    df = pd.read_csv(path)
    targets = []
    dates = []
    targets1 = []

    for index, row in df.iterrows():
        targets.append(row['ma5'])
        
        dates.append(row['date'])

        targets1.append(row['open'])

    if reverse:
        targets = targets[::-1]
        dates = dates[::-1]
        targets1 = targets1[::-1]

    slide_window = 15
    dayn = 1 #start from 0
    data = []
    label = []
    label_dates = []
    for i in range(len(dates) - slide_window - dayn):
        label.append(targets1[i + slide_window + dayn] - targets1[i + slide_window])
        label_dates.append(dates[i + slide_window])

    return np.array(label), np.array(label_dates)


for code in stock_codes:
    if str(code).startswith('3'):
        continue

    time.sleep(0.2)
    print('getting notices for %s'%code)

    y, dates = get_label_dates(code)
    dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    if len(dates) <= 10:
        print('no data for %s' % code)
        continue
 
    notes = get_notices_by_code(code, dates[pick_index])
    if notes is None or notes.empty:
        continue

    titles = ''
    for index, row in notes.iterrows():
        titles += row['title'].encode('utf-8') + ';'

    log.write('%s,%s,%s,%s\n'%(code, row['date'], y[pick_index], titles))

#update_netease_by_code(600082)

log.close()
