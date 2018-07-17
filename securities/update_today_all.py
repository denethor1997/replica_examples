import time
import os
import numpy as np
import pandas as pd
import tushare as ts
import datetime

save_dir = 'data/today_all/'
save_log = 'data/today_all/update.log'

def update_today_all():
    date = datetime.datetime.now().strftime('%Y-%m-%d')

    new_df = ts.get_today_all()
    csv_path = os.path.join(save_dir, str(date) + '.csv')

    if new_df is None:
        log.write('failed to load data:%s\n' % date)
        return

    print('new df shape:%s' % (new_df.shape,))

    new_df.to_csv(csv_path, encoding='utf-8')

update_today_all()

