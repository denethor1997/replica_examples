import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.netease import *
from datetime import datetime

save_dir = 'data/netease/hist'
save_log = 'data/netease/update.log'

"""
# 600004  603999   sh
stock_code_start_sh = 600004
stock_code_end_sh = 603999

# 000002  002815   sz
stock_code_start_sz = 2
stock_code_end_sz = 2815


stock_codes = [code for code in range(stock_code_start_sh, stock_code_end_sh)] #603996

stock_codes += [code for code in range(stock_code_start_sz, stock_code_end_sz)]
"""

today = datetime.now().strftime('%Y-%m-%d')
df = ts.get_day_all()

if df is None or df.empty:
    print('failed to get codes')
    exit(-1)

stock_codes = df['code'].tolist()
print('total codes:%s' % len(stock_codes))

log = open(save_log, 'w')

def update_netease_by_code(code, ktype='D'):
    old_csv_path = os.path.join(save_dir, str(code).zfill(6) + '_' +  ktype + '.csv')

    # if old csv not exist
    old_df = None
    old_date = None
    start_date = None
    if os.path.exists(old_csv_path):
        old_df = pd.read_csv(old_csv_path, index_col=['date'])
        if not old_df.empty:
            #old_df.set_index('date')
            old_date = old_df.index.values[0]
            start_date = old_date.replace('-','')
            print('old df shape:%s' % (old_df.shape,))
        else:
            old_df = None

    new_df = get_hist_netease(code=str(code), start_date=start_date)

    if new_df is None or new_df.empty:
        log.write('failed to load data:%s\n' % code)
        return

    print('new df shape:%s' % (new_df.shape,))
    if old_df is not None:
        #print(old_df.head(1))
        #print(new_df.head(1))
        old_df = old_df.drop([old_date])
        new_df = new_df.append(old_df)

    new_df.to_csv(old_csv_path, encoding='utf-8')
    log.write('load data done:%s\n' % code)

for code in stock_codes:
    time.sleep(0.3)   

    update_netease_by_code(code)

#update_netease_by_code(600082)

log.close()
