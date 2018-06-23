# -*- coding:utf-8 -*-
import os
import traceback
import time

import matplotlib.pyplot as plt
import numpy as np

#os.environ['KERAS_BACKEND'] = "theano"
#os.environ['THEANO_FLAGS'] = "device=cpu"
from data_preprocess.preprocess import *
from data_preprocess.load_data import *
import csv
from data_preprocess.Extract_Features import Extract_Features
from keras.models import model_from_json
from algorithm.reg_lstm import reg_lstm
from algorithm.rmse import *

import tushare as ts
# import myglobal
import shutil


dates = []
oneDayLine = []
thirtyDayLine = []
month_dates = []
acc_result = []

acc_result = []

#stock_code = '600169'
#stock_code = '600082'
stock_code = '600083'

download_economy()

models_path = './data/models_22_test/'

# 删除原有目录，××××注意××××××
# shutil.rmtree(models_path,True)
# os.remove('./models/*')  #清空
if os.path.isdir(models_path) is not True:
    os.mkdir(models_path)

stock_data_path = './data/stock_data/'
if os.path.isdir(stock_data_path) is not True:
    os.mkdir(stock_data_path)

# 
open_index_sh, close_index_sh, volume_index_sh, ma5_index_sh, vma5_index_sh, dates_index_sh = load_index_open_close_volume_ma5_vma5_from_tushare(
    stock_data_path + '../sh.csv')

# 
open_index_sz, close_index_sz, volume_index_sz, ma5_index_sz, vma5_index_sz, dates_index_sz = load_index_open_close_volume_ma5_vma5_from_tushare(
    stock_data_path + '../sz.csv')


def compute_code(code):
    time.sleep(0.1)  # 
    if len(str(code))<6:
        code = ''.join('0' for _ in range(6-len(str(code))))+str(code)
    try:
        # print "this is code ", code
        if download_fq_data_from_tushare(code):
            print code, "download over ~ "
        else:
            print('failed to download %s' % code)
            return
        download_from_tushare(code)
        # oneDayLine, dates = load_data_from_tushare(stock_data_path + str(code) + '.csv')
        # volume, volume_dates = load_volume_from_tushare(stock_data_path + str(code) + '.csv')
        open_price, oneDayLine, volume, ma5, vma5, turnover, dates = load_fq_open_close_volume_ma5_vma5_turnover_from_tushare(stock_data_path + str(code) + '_fq.csv')
        if (str(code)[0] == '6'):
            # 
            open_index, close_index, volume_index, ma5_index, vma5_index, dates_index = open_index_sh, close_index_sh, volume_index_sh, ma5_index_sh, vma5_index_sh, dates_index_sh
        else:
            # 
            open_index, close_index, volume_index, ma5_index, vma5_index, dates_index = open_index_sz, close_index_sz, volume_index_sz, ma5_index_sz, vma5_index_sz, dates_index_sz


        # thirtyDayLine, month_dates = load_data_from_tushare(stock_data_path + str(code) + '_month.csv')
        if len(oneDayLine) < 400:
            return
        ef = Extract_Features()
        daynum = 5
        '''
        ~~~~~ for classification ~~~~~~ X is delta close price, y is 10 for increase while 01 for decrease
        '''
        X_clf = []
        y_clf = []
        for i in range(daynum, len(oneDayLine)-1):
            #
            big_deals = 0 #get_big_deal_volume(code, dates[i])

            '''
            '''
            p = dates_index.index(dates[i])
            #

            X_delta = [oneDayLine[k] - oneDayLine[k - 1] for k in range(i - daynum, i)] + \
                      [volume[k] - volume[k-1] for k in range(i - daynum, i)] + \
                      [turnover[k] for k in range(i - daynum, i)] + \
                      [ma5[i]] + \
                      [vma5[i]] + \
                      [open_index[p]] + [close_index[p]] + [volume_index[p]] + [ma5_index[p]] + [vma5_index[p]] + \
                      [big_deals] + \
                      [ef.parse_weekday(dates[i])] + \
                      [ef.lunar_month(dates[i])] + \
                      [ef.MoneySupply(dates[i])]
                      # [ef.rrr(dates[i - 1])] + \
            X_clf.append(X_delta)
            y_clf.append(oneDayLine[i + 1] - oneDayLine[i])

        print(len(X_clf))
        print(X_clf[0])
        print(len(y_clf))
        print(y_clf[0])

        X_clf = preprocessing.MinMaxScaler().fit_transform(X_clf)
        y_clf = preprocessing.MinMaxScaler().fit_transform(np.reshape(np.array(y_clf), (-1, 1)))

        #!
        X_clf_train, X_clf_test, y_clf_train, y_clf_test = create_Xt_Yt(X_clf, y_clf, 0.86)#0.8
        X_clf_train = np.array(X_clf_train)
        X_clf_test = np.array(X_clf_test)
        X_clf_train = np.reshape(X_clf_train, (X_clf_train.shape[0], 1, X_clf_train.shape[1]))
        X_clf_test = np.reshape(X_clf_test, (X_clf_test.shape[0], 1, X_clf_test.shape[1]))
        input_dime = len(X_clf[0])
        # out = input_dime * 2 + 1
        if True:#not os.path.isfile('./data/model_'+str(code)+'.h5'):
            model = reg_lstm(input_dime)
            model.fit(X_clf_train,
                      np.array(y_clf_train),
                      nb_epoch=400,
                      batch_size=50,
                      verbose=0,
                      # validation_split=0.12
                      )
            """
            # serialize model to JSON
            model_json = model.to_json()
            with open(models_path + "model_" + str(code) + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(models_path +"model_" + str(code) + ".h5")
            print("Saved model to disk")
            """
        else:

            json_file = open(models_path + 'model_' + str(code) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(models_path + "model_" + str(code) + ".h5")
            print("Loaded model from disk")
            print "model" + str(code) + "loaded!"

        score = model.evaluate(X_clf_test, y_clf_test, batch_size=10)

        print "****************************************"
        print 'code =', code
        print "****************************************"

        print ""
        print "test : ", model.metrics_names[0], score[0], model.metrics_names[1], score[1]
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        predicted_y_clf_test = model.predict(X_clf_test)
        my_rmse = rmse(predicted_y_clf_test, y_clf_test)
        print("rmse = %s" % my_rmse)

        plt.figure(1)
        plt.plot(dates[len(dates)-len(y_clf_test):len(dates)], y_clf_test, color='g')
        plt.plot(dates[len(dates)-len(predicted_y_clf_test):len(dates)], predicted_y_clf_test, color='r')
        plt.show()

        predicted_y_clf_train = model.predict(X_clf_train)
        plt.figure(1)
        plt.plot(dates[0:len(y_clf_train)], y_clf_train, color='g')
        plt.plot(dates[0:len(y_clf_train)], predicted_y_clf_train, color='r')
        plt.show()

    except Exception as e:
        traceback.print_exc()
        # 
        print code, "is non type or is too less data!"
        return

compute_code(stock_code)


