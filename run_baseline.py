import argparse
import pdb
import json
import itertools

import torch
import numpy as np

from baseline.Bagel.model import DonutX, Donut, VAE
import pandas as pd
import numpy as np
from baseline.Bagel.kpi_series import KPISeries
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from baseline.Bagel.evaluation_metric import range_lift_with_delay

from baseline.SPOT.spot import SPOT, bidSPOT

from baseline.MicrosoftSR.msanomalydetector import SpectralResidual
from baseline.MicrosoftSR.msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW
from baseline.MicrosoftSR.srcnn.utils import sr_cnn, sr_cnn_eval
from baseline.MicrosoftSR.srcnn.generate_data import gen


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--model', type=str, choices=['bagel', 'donut', 'vae', 'spot', 'dspot', 'sr', 'srcnn'])
    args.add_argument('--data', type=str, default='./data/kpi/series_0_05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv')
    args.add_argument('--seed', type=int, default=2019)
    args.add_argument('--cpu', action='store_true')
    args.add_argument('--delay', type=int, default=7)
    args.add_argument('--label', type=float, default=0.0)

    # SPOT
    args.add_argument('--risk', type=float, default=None)
    args.add_argument('--depth', type=float, default=None)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    print('Dataset: %s...'%args.data)
    print('Setting manual seed %d...'%args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    df = pd.read_csv(args.data, header=0, index_col=None)
    kpi = KPISeries(
        value = df.value,
        timestamp = df.timestamp,
        label = df.label,
        name = 'sample_data',
    )

    # Splitting
    train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))

    if args.model == 'spot':
        q = args.risk

        init_data = train_kpi.value.reshape(-1)
        val_data = valid_kpi.value.reshape(-1)
        test_data = test_kpi.value.reshape(-1)

        if q is None:
            from threading import Thread

            print('Iterating parameter candidates...')

            q_candidates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
            y_prob = np.zeros((len(q_candidates), test_data.shape[0]))

            def run_spot(id, q):
                s = SPOT(q)  # SPOT object
                s.fit(np.concatenate((init_data, val_data)), test_data)  # data import
                s.initialize(verbose=False)  # initialization step
                results = s.run()  # run

                y_prob[id][results['alarms']] = 1

            tasks = []
            for id in q_candidates:
                th = Thread(target=run_spot, args=(id, q))
                th.start()
                tasks.append(th)

            for th in tasks:
                th.join()
        else:
            s = SPOT(q)  # SPOT object
            s.fit(np.concatenate((init_data, val_data)), test_data)  # data import
            s.initialize()  # initialization step
            results = s.run()  # run

            y_prob = np.zeros(test_data.shape[0])
            y_prob[results['alarms']] = 1
    elif args.model == 'dspot':
        q = args.risk
        d = args.depth

        init_data = train_kpi.value.reshape(-1)
        val_data = valid_kpi.value.reshape(-1)
        test_data = test_kpi.value.reshape(-1)

        if q is None:
            from threading import Thread

            print('Iterating parameter candidates...')

            q_candidates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
            d_candidates = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
            y_prob = np.zeros((len(q_candidates) * len(d_candidates), test_data.shape[0]))

            def run_spot(id, q, d):
                s = bidSPOT(q, d)
                s.fit(np.concatenate((init_data, val_data)), test_data)  # data import
                s.initialize(verbose=False)  # initialization step
                results = s.run()  # run

                y_prob[id][results['alarms']] = 1

            tasks = []
            for id, (q, d) in enumerate(itertools.product(q_candidates, d_candidates)):
                th = Thread(target=run_spot, args=(id, q, d))
                th.start()
                tasks.append(th)

            for th in tasks:
                th.join()
        else:
            s = bidSPOT(q, d)
            s.fit(np.concatenate((init_data, val_data)), test_data)  # data import
            s.initialize()  # initialization step
            results = s.run()  # run

            y_prob = np.zeros(test_data.shape[0])
            y_prob[results['alarms']] = 1
    elif args.model == 'sr':
        def detect_anomaly(series, threshold, mag_window, score_window):
            detector = SpectralResidual(series=series, threshold=threshold, mag_window=mag_window,
                                        score_window=score_window)
            return detector.detect()

        result = detect_anomaly(pd.DataFrame({'timestamp': test_kpi.timestamp, 'value': test_kpi.value}), THRESHOLD, MAG_WINDOW, SCORE_WINDOW)
        y_prob = result['isAnomaly'].values
    elif args.model == 'srcnn':
        # Configuration
        srcnn_window_size = 128
        srcnn_step = 64
        srcnn_num = 10
        srcnn_lr = 1e-6
        srcnn_epoch = 10
        srcnn_batch = 256
        srcnn_delay = 7
        srcnn_threshold = 0.95

        results = []
        print("generating train data")
        generator = gen(win_siz=srcnn_window_size, step=srcnn_step, nums=srcnn_num)
        train_timestamp = train_kpi.timestamp
        train_value = train_kpi.value

        train_data = generator.generate_train_data(train_value)
        results += train_data

        with open('./cache/train.json', 'w') as f:
            json.dump(results, f)

        sr_cnn(data_path='./cache/srcnn/train.json', model_path='./cache/srcnn/', win_size=srcnn_window_size, lr=srcnn_lr, epochs=srcnn_epoch, batch=srcnn_batch, num_worker=4, load_path=None)

    else:
        # Normalization
        train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
        valid_kpi = valid_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)
        test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

        if args.model == 'bagel':
            model = DonutX(cuda=(not args.cpu), max_epoch=50, latent_dims=8, network_size=[100, 100])
        elif args.model == 'donut':
            model = Donut(cuda=(not args.cpu), max_epoch=50, latent_dims=8, network_size=[100, 100])
        elif args.model == 'vae':
            model = VAE(cuda=(not args.cpu), max_epoch=50, latent_dims=8, network_size=[100, 100])
        else:
            raise ValueError('Invalid model!')
        model.fit(train_kpi.label_sampling(args.label), valid_kpi)
        y_prob = model.predict(test_kpi.label_sampling(0.))

    if (args.model == 'spot' or args.model == 'dspot') and args.risk is None:
        best_f1 = 0
        precision = 0
        recall = 0
        pr_auc = 0
        roc_auc = 0
        for y_pred in y_prob:
            y_pred = range_lift_with_delay(y_pred, test_kpi.label, delay=args.delay)
            precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_pred)

            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            inds = np.argsort(precisions)
            pr_auc = max(auc(precisions[inds], recalls[inds]), pr_auc)
            roc_auc = max(roc_auc_score(test_kpi.label, y_pred), roc_auc)
            best_f1_ind = np.argmax(f1_scores[np.isfinite(f1_scores)])
            best_f1 = max(np.max(f1_scores[np.isfinite(f1_scores)]), best_f1)
            precision = max(precisions[np.isfinite(f1_scores)][best_f1_ind], precision)
            recall = max(recalls[np.isfinite(f1_scores)][best_f1_ind], recall)
    else:
        y_prob = range_lift_with_delay(y_prob, test_kpi.label, delay=args.delay)
        precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_prob)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        inds = np.argsort(precisions)
        pr_auc = auc(precisions[inds], recalls[inds])
        roc_auc = roc_auc_score(test_kpi.label, y_prob)
        best_f1_ind = np.argmax(f1_scores[np.isfinite(f1_scores)])
        best_f1 = np.max(f1_scores[np.isfinite(f1_scores)])
        precision = precisions[np.isfinite(f1_scores)][best_f1_ind]
        recall = recalls[np.isfinite(f1_scores)][best_f1_ind]

    result_df = pd.DataFrame({'F1': [best_f1], 'Precision': [precision], 'Recall': [recall],
                              'PR': [pr_auc], 'ROC': [roc_auc]})
    info_df = pd.DataFrame({'Model': [args.model], 'Data': [args.data.split('/')[-1]], 'label': [args.label], 'seed': [args.seed]})
    # print('best F1-score: %f'%np.max(f1_scores[np.isfinite(f1_scores)]))
    # print('PR_AUC: %f'%pr_auc)
    # print('ROC_AUC: %f'%roc_auc)
    print(info_df)
    print(result_df)
