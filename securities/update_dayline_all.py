import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime

save_dir = 'data/dayline/latest'
bak_dir = 'data/dayline/bak'
save_log = 'data/dayline/update.log'

"""
# 600004  603999   sh
stock_code_start_sh = 600004
stock_code_end_sh = 603999

# 000002  002815   sz
stock_code_start_sz = 2
stock_code_end_sz = 2815

#download_economy()

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

def update_dayline_by_code(code, ktype='D'):
    old_csv_path = os.path.join(save_dir, str(code) + '_' +  ktype + '.csv')

    # if old csv not exist
    old_df = None
    old_date = None
    if os.path.exists(old_csv_path):
        old_df = pd.read_csv(old_csv_path, index_col=['date'])
        #old_df.set_index('date')
        old_date = old_df.index.values[0]
        print(old_date)
        print('old df shape:%s' % (old_df.shape,))

    new_df = ts.get_hist_data(code=str(code), start=old_date, ktype=ktype)

    if new_df is None:
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
    #time.sleep(0.1)   
    if len(str(code))<6:
        code = ''.join('0' for _ in range(6-len(str(code))))+str(code)

    update_dayline_by_code(code)
    update_dayline_by_code(code, ktype='60')
    update_dayline_by_code(code, ktype='30')
    update_dayline_by_code(code, ktype='15')
    update_dayline_by_code(code, ktype='5')


#update_dayline_by_code(600082,ktype='D')

log.close()
