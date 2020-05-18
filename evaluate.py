import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from torch.utils.data import DataLoader
from tqdm import tqdm

from salad.dataset import prepare_dataset, KPIBatchedWindowDataset
from salad.metrics import modified_f1, modified_recall, modified_precision
from salad.misc import print_blue_info
from salad.model import DenseEncoder, DenseDecoder, ConvEncoder, ConvDecoder, DataDiscriminator, LatentDiscriminator
from salad.trainer import Trainer


##########################################################################################
# Argparse
##########################################################################################
def arg_parse():
    parser = argparse.ArgumentParser(description='SALAD: KPI Anomaly Detection')

    # Dataset
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument("--data", dest='data_path', type=str,
                               default='./data/kpi/series_0_05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv',
                               help='The dataset path')
    group_dataset.add_argument("--category", dest='data_category', choices=['kpi', 'nab', 'yahoo'], type=str,
                               required=True)
    group_dataset.add_argument("--split", dest='train_val_test_split', type=tuple, default=(5, 2, 3),
                               help='The ratio of train, validation, test dataset')
    group_dataset.add_argument("--filling", dest='filling_method', choices=['zero', 'linear'], default='zero')
    group_dataset.add_argument("--standardize", dest='standardization_method',
                               choices=['standrad', 'minmax', 'negpos1'],
                               default='negpos1')

    # Model
    group_model = parser.add_argument_group('Model')
    group_model.add_argument("--var", dest='variant', type=str, choices=['conv', 'dense', 'test'], default='conv')
    group_model.add_argument("--window", dest='window_size', type=int, default=128)
    group_model.add_argument("--hidden", dest='hidden_size', type=int, default=100)
    group_model.add_argument("--latent", dest='latent_size', type=int, default=16)

    # Save and load
    group_save_load = parser.add_argument_group('Save and Load')
    group_save_load.add_argument("--save", dest='save_path', type=str, default='./cache/uncategorized/')
    group_save_load.add_argument("--load", dest='load_epoch', type=int, required=True)

    # Devices
    group_device = parser.add_argument_group('Device')
    group_device.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=1, type=int)
    group_device.add_argument("--seed", dest='seed', type=int, default=2019, help="The random seed")

    # Training
    group_training = parser.add_argument_group('Training')
    group_training.add_argument("--epochs", dest="epochs", type=int, default=100, help="The number of epochs to run")
    group_training.add_argument("--label-portion", dest="label_portion", type=float, default=0.0,
                                help='The portion of labels used in training')
    group_training.add_argument("--batch", dest="batch_size", type=int, default=512, help="The batch size")

    # Detection
    group_detection = parser.add_argument_group('Detection')
    group_detection.add_argument("--delay", dest="delay", type=int, default=7, help='The delay of tolerance')
    group_detection.add_argument("--threshold", type=float, default=None,
                                 help='The threshold for determining anomalies')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    # GPU setting
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(-1 * np.array(gpu_memory))
    os.system('rm tmp')
    assert (args.num_gpu <= len(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids[:args.num_gpu]))
    print_blue_info('Current GPU [%s], free memory: [%s] MB' % (
        os.environ['CUDA_VISIBLE_DEVICES'], ','.join(map(str, np.array(gpu_memory)[gpu_ids[:args.num_gpu]]))))

    # Set the random seed
    if args.seed is not None:
        print_blue_info('Setting manual seed %d...' % args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Reading dataset
    print_blue_info('Reading dataset...')
    train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m = prepare_dataset(
        args.data_path, args.data_category, args.train_val_test_split,
        args.label_portion, args.standardization_method, args.filling_method)

    # Training dataset
    test_dataset = KPIBatchedWindowDataset(test_x, label=test_y, mask=test_m, window_size=args.window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False)

    # Models
    if args.variant == 'conv':
        encoder = ConvEncoder(args.window_size, 1, args.latent_size).cuda()
        decoder = ConvDecoder(args.window_size, 1, args.latent_size).cuda()
    elif args.variant == 'dense':
        encoder = DenseEncoder(args.window_size, args.hidden_size, args.latent_size).cuda()
        decoder = DenseDecoder(args.window_size, args.hidden_size, args.latent_size).cuda()
    else:
        raise ValueError('Invalid model variant!')
    data_discriminator = DataDiscriminator(args.window_size, args.hidden_size)
    latent_discriminator = LatentDiscriminator(args.hidden_size, args.latent_size)

    check_point = torch.load(args.save_path + 'model_epoch_%d.ckpt' % args.load_epoch)
    print_blue_info('Resume at epoch %d...' % check_point['epoch'])
    encoder.load_state_dict(check_point['encoder_state_dict'])
    decoder.load_state_dict(check_point['decoder_state_dict'])
    data_discriminator.load_state_dict(check_point['data_discriminator_state_dict'])
    latent_discriminator.load_state_dict(check_point['latent_discriminator_state_dict'])

    encoder.cuda()
    decoder.cuda()
    data_discriminator.cuda()
    latent_discriminator.cuda()

    # Define trainer
    trainer = Trainer(encoder, decoder, data_discriminator, latent_discriminator, batch_size=args.batch_size,
                      window_size=args.window_size)

    # Evaluating
    print_blue_info('Start evaluating...')

    y_pred = []
    y_true = []

    trainer.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(test_loader)):
            x, y, m = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()

            x_rec = trainer.reconstruct(x)
            data_rec_loss = nn.MSELoss(reduction='none')(x_rec, x)
            y_pred.append(data_rec_loss[:, -1].cpu().numpy().reshape(-1))
            y_true.append(y[:, -1].cpu().numpy().reshape(-1))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, y_pred)
    inds = np.argsort(pr_precisions)
    pr_auc = auc(pr_precisions[inds], pr_recalls[inds])
    roc_auc = roc_auc_score(y_true, y_pred)

    if args.threshold is None:
        print_blue_info('Using brute force search for the proper threshold...')
        candidate_values = np.concatenate(
            [y_pred[y_true == 0], np.sort(y_pred[y_true == 1])[:y_pred[y_true == 1].shape[0] // 5]], axis=0)
        candidates = np.linspace(np.min(y_pred[y_true == 0]), np.max(candidate_values), 1000)

        f1s = np.zeros_like(candidates)
        precisions = np.zeros_like(candidates)
        recalls = np.zeros_like(candidates)


        def calc_metric(th, num):
            y_res = np.zeros_like(y_pred)
            y_res[y_pred >= th] = 1.0

            f1 = modified_f1(y_pred=y_res, y_true=y_true, mask=test_m[args.window_size - 1:], delay=args.delay)
            precision = modified_precision(y_pred=y_res, y_true=y_true, mask=test_m[args.window_size - 1:],
                                           delay=args.delay)
            recall = modified_recall(y_pred=y_res, y_true=y_true, mask=test_m[args.window_size - 1:], delay=args.delay)

            f1s[num] = f1
            precisions[num] = precision
            recalls[num] = recall


        from threading import Thread

        tasks = []
        for i in tqdm(range(len(candidates))):
            th = Thread(target=calc_metric, args=(candidates[i], i))
            th.start()
            tasks.append(th)

        for th in tasks:
            th.join()

        best_f1_ind = np.argmax(f1s)

        print_blue_info('Best F1: %f' % f1s[best_f1_ind])
        print_blue_info('Precision: %f' % precisions[best_f1_ind])
        print_blue_info('Recall: %f' % recalls[best_f1_ind])
        print_blue_info('PR: %f' % pr_auc)
        print_blue_info('ROC: %f' % roc_auc)
    else:
        y_pred[y_pred < args.threshold] = 0
        y_pred[y_pred >= args.threshold] = 1

        f1 = modified_f1(y_pred=y_pred, y_true=y_true, mask=test_m[args.window_size - 1:], delay=args.delay)
        precision = modified_precision(y_pred=y_pred, y_true=y_true, mask=test_m[args.window_size - 1:],
                                       delay=args.delay)
        recall = modified_recall(y_pred=y_pred, y_true=y_true, mask=test_m[args.window_size - 1:], delay=args.delay)

        print_blue_info('F1: %f' % f1)
        print_blue_info('Precision: %f' % precision)
        print_blue_info('Recall: %f' % recall)
        print_blue_info('PR: %f' % pr_auc)
        print_blue_info('ROC: %f' % roc_auc)
