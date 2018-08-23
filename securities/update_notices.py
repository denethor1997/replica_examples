import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.netease import *
from datetime import datetime

stock_codes = [600082, 600169, 600036, '000999']

today = datetime.now().strftime('%Y-%m-%d')


df = ts.get_day_all()

if df is None or df.empty:
    print('failed to get codes')
    exit(-1)

stock_codes = df['code'].tolist()
print('total codes:%s' % len(stock_codes))

timestr = datetime.now().strftime('%Y-%m-%d')
log_path = os.path.join('snapshots_pick', 'update_notices_%s.log'%timestr)
log = open(log_path, 'w')

def get_notices_by_code(code):
    try:
        notices = ts.get_notices(code=str(code), date=today)
        return notices
    except Exception as e:
        print(e)
        return None

for code in stock_codes:
    time.sleep(0.2)
    print('getting notices for %s'%code)

    notes = get_notices_by_code(code)
    if notes is None or notes.empty:
        continue

    for index, row in notes.iterrows():
        log.write('%s,%s,%s\n'%(code, row['date'], row['title'].encode('utf-8')))

#update_netease_by_code(600082)

log.close()
