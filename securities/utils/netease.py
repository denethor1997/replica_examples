from __future__ import division

import time
import json
import pandas as pd
from pandas.compat import StringIO
import numpy as np
import os
from urllib2 import urlopen, Request
from datetime import datetime

def get_hist_netease(code, start_date='', end_date=''):

    code = str(code)

    if len(code) <= 6:
        code = code.zfill(6)
        if code[0] == '0' or code[0] == '3' or code[0] == '2':
            code = '1' + code
        else:
            code = '0' + code

    if start_date == '' or start_date == None:
        start_date = '19951218'

    if end_date == '' or end_date == None:
        end_date = datetime.now().strftime('%Y%m%d')

    print('loading from %s to %s:%s' % (start_date, end_date, code))
    url = 'http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'%(code, start_date, end_date)
    try:
        request = Request(url)
        lines = urlopen(request, timeout = 10).read()
        csv_str = lines.decode('gbk').encode('utf-8')
        df = pd.read_csv(StringIO(csv_str))
        df.columns = ['date','code','name','close','high','low','open','pre_close','price_change','p_change','turnover','volume','amount','mktcap','nmc']
        df = df.drop('name', axis=1)
        df = df.set_index('date')
        return df
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    code = '300104'
    df = get_hist_netease(code)
    if df is None:
        print('failed to load csv')
        exit(-1)

    #df.reset_index(drop=True, inplace=True)

    print(df.tail())
    df.to_csv(code + '_D.csv', encoding='utf-8')
