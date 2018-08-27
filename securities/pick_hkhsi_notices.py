import time
import os
import numpy as np
import pandas as pd
import tushare as ts
from utils.netease import *
from datetime import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hkhsi', type=str, required=True)
args = parser.parse_args()


today = datetime.now().strftime('%Y-%m-%d')


timestr = str(datetime.now()).replace(' ', '@').replace(':', '_')
log_path = os.path.join('snapshots_pick', 'pick_notices_%s.log'%timestr)
log = open(log_path, 'w')

def get_notices_by_code(code):
    try:
        notices = ts.get_notices(code=str(code), date=today)
        return notices
    except Exception as e:
        print(e)
        return None

start_time = time.time()
for line in open(args.hkhsi):
    tokens = line.strip().split('@')
    code = tokens[0]
    score = tokens[1]
    date = tokens[2]

    if str(code).startswith('3'):
        continue

    time.sleep(0.2)
    print('getting notices for %s'%code)

    notes = get_notices_by_code(code)
    if notes is None or notes.empty:
        continue

    titles = ''
    for index, row in notes.iterrows():
        titles += row['title'].encode('utf-8') + ';'

    log.write('%s,%s,%s,%s\n'%(code, score, date, titles))

#update_netease_by_code(600082)

log.close()

print("================================used time:%s secs=============================="%(time.time()-start_time))
print("log_path:%s"%log_path)
