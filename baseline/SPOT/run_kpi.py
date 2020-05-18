import pdb
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from spot import bidSPOT


file_list = os.listdir('./data/')
for file in file_list:
    print(file)
    df = pd.read_csv('./data/' + file)
    print(df.head)
    print(df.shape)


    value = df.value.values.reshape(-1)
    label = df.label.values.reshape(-1)

    n_init = df.shape[0]//20
    init_data = value[:n_init] 	# initial batch
    data = value[n_init:]  		# stream

    mode = 'vis' # ('search', 'vis')

    if mode == 'search':
        qs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ds = [5, 10, 50, 100, 500, 1000, 2000]

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        parameters = []

        for q in qs:
            for d in ds:
                s = bidSPOT(q, d)     	# biDSPOT object
                s.fit(init_data, data) 	# data import
                s.initialize() 	  		# initialization step
                results = s.run()    	# run

                y_pred = np.zeros(data.shape[0])
                y_pred[results['alarms']] = 1
                y_true = label[n_init:]

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                parameters.append((q, d))

        ind = np.argmax(f1_list)

        print('accuracy: ', accuracy_list[ind])
        print('precision: ', precision_list[ind])
        print('recall: ', recall_list[ind])
        print('f1: ', f1_list[ind])
        print('para: ', parameters[ind])
    else:
        # q = 1e-3 				# risk parameter
        # d = 2000  				# depth parameter

        q = 1e-3  # risk parameter
        d = 100  # depth parameter

        s = bidSPOT(q, d)  # biDSPOT object
        s.fit(init_data, data)  # data import
        s.initialize()  # initialization step
        results = s.run()  # run

        y_pred = np.zeros(data.shape[0])
        y_pred[results['alarms']] = 1
        y_true = label[n_init:]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print('accuracy: ', accuracy)
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1: ', f1)

        with open('./output/result.txt', 'wa') as f:
            f.write('------- ' + file + ' --------\n')
            f.write('accuracy: %f\n'%accuracy)
            f.write('precision: %f\n' % precision)
            f.write('recall: %f\n' % recall)
            f.write('f1: %f\n' % f1)

        plt.figure(figsize=(15,6))
        fig = s.plot(results, label=label[n_init:], show=False, save='./output/' + file + '.svg') 	 	# plot
        # plt.close(fig)
