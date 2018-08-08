from __future__ import division

import time
import json
import pandas as pd
from pandas.compat import StringIO
import numpy as np
import os
from urllib2 import urlopen, Request
from datetime import datetime

def get_hist_hkhsi(start_date='', end_date=''):


    if start_date == '' or start_date == None:
        start_date = '1995-12-18'

    if end_date == '' or end_date == None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print('loading from %s to %s' % (start_date, end_date))
    url = 'http://web.ifzq.gtimg.cn/appstock/app/kline/kline?_var=kline_day&param=%s,day,%s,%s,16400,'%('hkHSI', start_date, end_date)
    try:
        request = Request(url)
        lines = urlopen(request, timeout = 10).read()
        lines = lines.split('=')[1]
        js = json.loads(lines)
        klines = js['data']['hkHSI']['day']
        klines = klines[::-1]
        df = pd.DataFrame(klines, columns=['date','open','close','high','low','volume'])
        df = df.set_index('date')

        for col in df.columns[1:6]:
            df[col] = df[col].astype(float)

        last_vol = 0.0
        for index, row in df.iterrows():
            if row['volume'] - 0 < 0.0001:
                df.loc[index, 'volume'] = last_vol
            else:
                last_vol = row.volume

        return df 
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    df = get_hist_hkhsi()
    if df is None:
        print('failed to load csv')
        exit(-1)

    #df.reset_index(drop=True, inplace=True)

    print(df.tail())
    df.to_csv('hkHSI_D.csv', encoding='utf-8')
