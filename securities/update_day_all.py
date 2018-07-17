import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta, date

save_dir = 'data/day_all/latest'
save_log = 'data/day_all/update.log'

log = open(save_log, 'w')
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def update_day_all(date):

    try:
        new_df = ts.get_day_all(date)
    except:
        log.write('failed to call ts:%s\n' % date)
        return

    csv_path = os.path.join(save_dir, str(date) + '.csv')

    if new_df is None:
        log.write('failed to load data:%s\n' % date)
        return

    print('new df shape:%s' % (new_df.shape,))

    new_df.to_csv(csv_path, encoding='utf-8')

start_date = date(2017, 05, 01)

old_csvs = os.listdir(save_dir)
if (old_csvs is not None) and (len(old_csvs) > 0):
    old_csvs.sort(reverse=True)
    old_latest_csv = old_csvs[0]
    old_latest = os.path.splitext(old_latest_csv)[0]
    old_tokens = old_latest.split('-')
    start_date = date(int(old_tokens[0]), int(old_tokens[1]), int(old_tokens[2]))

now = datetime.now()
end_date = date(now.year, now.month, now.day)

print('from %s to %s' % (start_date, end_date))

for single_date in daterange(start_date, end_date):
    date_str = single_date.strftime('%Y-%m-%d')
    update_day_all(date_str)

