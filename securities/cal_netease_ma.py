import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.netease import *

read_dir = 'data/netease/hist'
write_dir = 'data/netease/hist_ma'

def cal_netease_ma(csv_path):
    if not os.path.exists(csv_path):
        print('invalid path:%s' % csv_path)
        return None

    df = pd.read_csv(csv_path, index_col=['date'])
    if df.empty:
        print('empty csv:%s' % csv_path)
        return None

    df = df.loc[df['close'] > 0.01]

    df_reverse = df[::-1]
    df['ma5'] = df_reverse['close'].rolling(5,min_periods=1).mean()[::-1]
    df['ma10'] = df_reverse['close'].rolling(10,min_periods=1).mean()[::-1]
    df['ma20'] = df_reverse['close'].rolling(20,min_periods=1).mean()[::-1]
    df['v_ma5'] = df_reverse['volume'].rolling(5,min_periods=1).mean()[::-1]
    df['v_ma10'] = df_reverse['volume'].rolling(10,min_periods=1).mean()[::-1]
    df['v_ma20'] = df_reverse['volume'].rolling(20,min_periods=1).mean()[::-1]
    return df


if __name__ == '__main__':
    for path in os.listdir(read_dir):
        print('processing:%s' % path)
        csv_path = os.path.join(read_dir, path)
        df = cal_netease_ma(csv_path)

        new_csv_path = os.path.join(write_dir, path)
        df.to_csv(new_csv_path, encoding='utf-8')
