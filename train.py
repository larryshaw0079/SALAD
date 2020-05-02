import os
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from salad.dataset import prepare_dataset, KPIBatchedWindowDataset
from salad.model import DenseEncoder, DenseDecoder, ConvEncoder, ConvDecoder, DataDiscriminator, LatentDiscriminator
from salad.trainer import Trainer
from salad.misc import print_blue_info, print_red_info


##########################################################################################
# Argparse
##################################################### #####################################
def arg_parse():
    parser = argparse.ArgumentParser(description='SALAD: KPI Anomaly Detection')

    # Dataset
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument("--data", dest='data_path', type=str,
                        default='./data/kpi/series_0_05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv', help='The dataset path')
    group_dataset.add_argument("--category", dest='data_category', choices=['kpi', 'nab', 'yahoo'], type=str, required=True)
    group_dataset.add_argument("--split", dest='train_val_test_split', type=tuple, default=(5, 2, 3),
                        help='The ratio of train, validation, test dataset')
    group_dataset.add_argument("--filling", dest='filling_method', choices=['zero', 'linear'], default='zero')
    group_dataset.add_argument("--standardize", dest='standardization_method', choices=['standrad', 'minmax', 'negpos1'],
                        default='negpos1')

    # Model
    group_model = parser.add_argument_group('Model')
    group_model.add_argument("--print-model", dest='print_model', action='store_true')
    group_model.add_argument("--var", dest='variant', type=str, choices=['conv', 'dense', 'test'], default='conv')
    group_model.add_argument("--window", dest='window_size', type=int, default=128)
    group_model.add_argument("--hidden", dest='hidden_size', type=int, default=100)
    group_model.add_argument("--latent", dest='latent_size', type=int, default=16)
    group_model.add_argument("--gen-lr", dest='gen_lr', type=float, default=1e-3)
    group_model.add_argument("--dis-lr", dest='dis_lr', type=float, default=1e-4)

    group_model.add_argument("--critic", dest='critic_iter', type=int, default=2)
    group_model.add_argument("--rec-weight", dest='rec_weight', type=float, default=1.0)
    group_model.add_argument("--contras", dest='use_contrastive', action='store_true')
    group_model.add_argument("--itimp", dest='in_train_imputation', action='store_true')
    group_model.add_argument("--margin", dest='contrastive_margin', type=float, default=1.0)

    # Save and load
    group_save_load = parser.add_argument_group('Save and Load')
    group_save_load.add_argument("--resume", dest='resume', action='store_true')
    group_save_load.add_argument("--save", dest='save_path', type=str, default='./cache/uncategorized/')
    group_save_load.add_argument("--interval", dest='save_interval', type=int, default=10)

    # Devices
    group_device = parser.add_argument_group('Device')
    group_device.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=1, type=int)
    group_device.add_argument("--seed", dest='seed', type=int, default=2019, help="The random seed")

    # Training
    group_training = parser.add_argument_group('Training')
    group_training.add_argument("--epochs", dest="epochs", type=int, default=150, help="The number of epochs to run")
    group_training.add_argument("--batch", dest="batch_size", type=int, default=512, help="The batch size")
    group_training.add_argument("--label-portion", dest="label_portion", type=float, default=0.0,
                        help='The portion of labels used in training')

    return parser.parse_args()


if __name__ == '__main__':
    # Argument parse
    args = arg_parse()
    print_blue_info('Arguments parsed...')
    print(args)

    # GPU setting
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(-1 * np.array(gpu_memory))
    os.system('rm tmp')
    assert (args.num_gpu <= len(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids[:args.num_gpu]))
    print_blue_info('Current GPU [%s], free memory: [%s] MB'% (
        os.environ['CUDA_VISIBLE_DEVICES'], ','.join(map(str, np.array(gpu_memory)[gpu_ids[:args.num_gpu]]))))

    # Preparing
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
    train_dataset = KPIBatchedWindowDataset(train_x, label=train_y, mask=train_m, window_size=args.window_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Validation dataset
    val_dataset = KPIBatchedWindowDataset(val_x, label=val_y, mask=val_m, window_size=args.window_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Models
    if args.variant == 'conv':
        encoder = ConvEncoder(args.window_size, 1, args.latent_size).cuda()
        decoder = ConvDecoder(args.window_size, 1, args.latent_size).cuda()
    elif args.variant == 'dense':
        encoder = DenseEncoder(args.window_size, args.hidden_size, args.latent_size).cuda()
        decoder = DenseDecoder(args.window_size, args.hidden_size, args.latent_size).cuda()
    else:
        raise ValueError('Invalid model variant!')
    data_discriminator = DataDiscriminator(args.window_size, args.hidden_size).cuda()
    latent_discriminator = LatentDiscriminator(args.hidden_size, args.latent_size).cuda()

    # Load models
    if args.resume:
        check_point = torch.load(args.save_path)
        print_blue_info('Resume at epoch %d...' % check_point['epoch'])
        encoder.load_state_dict(check_point['encoder_state_dict'])
        decoder.load_state_dict(check_point['decoder_state_dict'])
        data_discriminator.load_state_dict(check_point['data_discriminator_state_dict'])
        latent_discriminator.load_state_dict(check_point['latent_discriminator_state_dict'])

    # Define trainer
    trainer = Trainer(encoder, decoder, data_discriminator, latent_discriminator, args.batch_size,
                      args.window_size)
    if args.print_model:
        trainer.print_model()

    # Optimizers
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=2.5 * 1e-5)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=2.5 * 1e-5)
    optimizer_ddis = torch.optim.Adam(data_discriminator.parameters(), lr=args.dis_lr, betas=(0.5, 0.999),
                                      weight_decay=2.5 * 1e-5)
    optimizer_ldis = torch.optim.Adam(latent_discriminator.parameters(), lr=args.dis_lr, betas=(0.5, 0.999),
                                      weight_decay=2.5 * 1e-5)

    # Schedulers
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=10, gamma=0.75)
    scheduler_dec = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=10, gamma=0.75)
    scheduler_ddis = torch.optim.lr_scheduler.StepLR(optimizer_ddis, step_size=10, gamma=0.75)
    scheduler_ldis = torch.optim.lr_scheduler.StepLR(optimizer_ldis, step_size=10, gamma=0.75)

    # torch.autograd.set_detect_anomaly(True)

    trainer.train()
    # Train epoch
    for epoch in range(args.epochs):
        # Train batch
        for i, inputs in enumerate(tqdm(train_loader, desc='EPOCH: [%d/%d]' % (epoch + 1, args.epochs))):
            x, y, m = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()

            if args.in_train_imputation and epoch > args.epochs//3:
                with torch.no_grad():
                    for mcmc_iter in range(5):
                        x[m == 1] = (trainer.reconstruct(x) * ((epoch + 1) / args.epochs))[m == 1]

            ##################################################################################
            # Data discrimination
            ##################################################################################
            for k in range(args.critic_iter):
                data_dis_loss = trainer.data_dis_loss(x)
                data_discriminator.zero_grad()
                data_dis_loss.backward()
                optimizer_ddis.step()

            if args.use_contrastive:
                data_gen_loss, data_rec_loss = trainer.data_gen_loss(x, y=y, margin=args.contrastive_margin)
            else:
                data_gen_loss, data_rec_loss = trainer.data_gen_loss(x)
            data_loss = data_gen_loss + args.rec_weight * data_rec_loss
            encoder.zero_grad()
            decoder.zero_grad()
            data_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            ##################################################################################
            # Latent discrimination
            ##################################################################################

            # Latent discriminator
            for k in range(args.critic_iter):
                latent_dis_loss = trainer.latent_dis_loss(x)
                latent_discriminator.zero_grad()
                latent_dis_loss.backward()
                optimizer_ldis.step()

            # Generator
            latent_gen_loss = trainer.latent_gen_loss(x)
            encoder.zero_grad()
            latent_gen_loss.backward()
            optimizer_enc.step()

        # Learning rate adjustment
        scheduler_enc.step()
        scheduler_dec.step()
        scheduler_ddis.step()
        scheduler_ldis.step()

        # Save state dicts
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'data_discriminator_state_dict': data_discriminator.state_dict(),
                'latent_discriminator_state_dict': latent_discriminator.state_dict()
            }, args.save_path + 'model_epoch_%d.ckpt' % (epoch + 1))
