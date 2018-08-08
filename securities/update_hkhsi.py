import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.tt import *
from datetime import datetime

save_dir = 'data'

today = datetime.now().strftime('%Y-%m-%d')

def update_hkhsi():
    old_csv_path = os.path.join(save_dir, 'hkHSI_D.csv')

    # if old csv not exist
    old_df = None
    old_date = None
    start_date = None
    if os.path.exists(old_csv_path):
        old_df = pd.read_csv(old_csv_path, index_col=['date'])
        if not old_df.empty:
            #old_df.set_index('date')
            old_date = old_df.index.values[0]
            start_date = old_date
            print('old df shape:%s' % (old_df.shape,))
        else:
            old_df = None

    new_df = get_hist_hkhsi(start_date=start_date)

    if new_df is None or new_df.empty:
        print('failed to load hkHSI data')
        return

    print('new df shape:%s' % (new_df.shape,))
    if old_df is not None:
        #print(old_df.head(1))
        #print(new_df.head(1))
        old_df = old_df.drop([old_date])
        new_df = new_df.append(old_df)

    new_df.to_csv(old_csv_path, encoding='utf-8')

update_hkhsi()

