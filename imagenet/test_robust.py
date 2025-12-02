import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
import torch.nn.functional as F
# from spikingjelly.activation_based import functional, surrogate
from spikingjelly.clock_driven import functional
import models.sew_resnet as sew_resnet
import utils
from models.Resnet_ann import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,5'


_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

import numpy as np
np.random.seed(_seed_)

# 评估模型在测试集上的性能，计算损失和准确率（Top-1 和 Top-5）
def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output , _ = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)
            if output.dim() == 3:
                with torch.no_grad():
                    output = output.mean(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5

# def _get_cache_path(filepath):
#     import hashlib
#     h = hashlib.sha1(filepath.encode()).hexdigest()
#     cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
#     cache_path = os.path.expanduser(cache_path)
#     print(f"[DEBUG] 缓存路径: {cache_path}")  #
#     return cache_path
#
#
# def load_data(traindir, valdir, cache_dataset, distributed):
#     # Data loading code
#     print("Loading data")
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#     print("Loading training data")
#     st = time.time()
#     # 生成预处理标识符，包含RandomErasing参数
#     re_params = "re_p0.5_scale0.02-0.33_ratio0.3-3.3"
#     modified_traindir = f"{traindir}_{re_params}"
#     cache_path = _get_cache_path(modified_traindir)  # 使用修改后的traindir
#     #cache_path = _get_cache_path(traindir)
#     if cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print("Loading dataset_train from {}".format(cache_path))
#         dataset, _ = torch.load(cache_path)
#     else:
#         print("Creating new dataset with RandomErasing")
#         dataset = torchvision.datasets.ImageFolder(
#             traindir,
#             transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
#                 normalize,
#             ]))
#
#         # # 创建保存目录
#         # save_dir = "/zjh/ssj/Reg_ipmsn/imagenet_ipmsn/img_random_erasing"
#         # os.makedirs(save_dir, exist_ok=True)
#         #
#         # # 反标准化函数
#         # def denormalize(tensor):
#         #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#         #     return tensor * std + mean  # 逆标准化计算
#         #
#         # # 获取并保存单张图像
#         # image, label = dataset[0]
#         # denorm_img = denormalize(image)  # 反标准化
#         # denorm_img = denorm_img.permute(1, 2, 0).numpy()  # [H, W, C]
#         #
#         # plt.imshow(denorm_img)
#         # plt.axis('off')
#         # plt.savefig(f"{save_dir}/aug_sample_1.png", bbox_inches='tight', pad_inches=0)
#         # plt.close()
#
#         if cache_dataset:
#             print(f"Saving dataset_train to {cache_path}")
#             utils.mkdir(os.path.dirname(cache_path))
#             utils.save_on_master((dataset, traindir), cache_path)
#
#     print("Took", time.time() - st)
#
#     print("Loading validation data")
#     cache_path = _get_cache_path(valdir)
#     if cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print("Loading dataset_test from {}".format(cache_path))
#         dataset_test, _ = torch.load(cache_path)
#     else:
#         dataset_test = torchvision.datasets.ImageFolder(
#             valdir,
#             transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize,
#             ]))
#         if cache_dataset:
#             print("Saving dataset_test to {}".format(cache_path))
#             utils.mkdir(os.path.dirname(cache_path))
#             utils.save_on_master((dataset_test, valdir), cache_path)
#
#     print("Creating data loaders")
#     if distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
#     else:
#         train_sampler = torch.utils.data.RandomSampler(dataset)
#         test_sampler = torch.utils.data.SequentialSampler(dataset_test)
#
#     return dataset, dataset_test, train_sampler, test_sampler

# 缓存文件的路径
def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

# 加载训练集和测试集数据，并根据是否使用分布式训练选择合适的数据采样器。
def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    print('gpu used:', torch.cuda.device_count())

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    # TensorBoard写入器对象
    train_tb_writer = None
    te_tb_writer = None

    # 初始化分布式模式，以便在多台机器或多个GPU上并行处理任务
    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.cos_lr_T == -1:
        args.cos_lr_T = args.epochs

    output_dir += f'_coslr{args.cos_lr_T}'

    if args.adamw:
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    output_dir += f'_{args.world_size}gpu'

    if args.load is not None:
        output_dir += '_load'

    if args.tet:
        output_dir += '_tet'


    if output_dir:
        utils.mkdir(output_dir)


    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")

    if args.model in sew_resnet.__dict__: # args.model = resnet18
        # model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', spiking_neuron=PMSNSJ,
        #                                         surrogate_function=surrogate.ATan(), T=args.T)
        # model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', spiking_neuron=PMSNSJ,
        #                                         T=args.T)
        # model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', m_weight=args.m_weight,
        #                                         spiking_neuron=iPMSN,
        #                                         T=args.T)
        model = sew_resnet.__dict__[args.model](zero_init_residual=False, T=args.T,
                                                connect_f='ADD')
    else:
        raise NotImplementedError(args.model)

    # print(model)

    if args.load is not None:
        model.load_state_dict(torch.load(args.load), strict=False)
        print('load', args.load)

    model.cuda()

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.tet:
        criterion = functional.temporal_efficient_training_cross_entropy
    else:
        def ce_loss(y, target):
            return F.cross_entropy(y.mean(0), target)
        criterion = ce_loss


    if args.adamw:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)
    

    model_without_ddp = model

    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DataParallel(model)
        model = model.cuda()
        model_without_ddp = model.module


    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

        # 打印加载的检查点信息
        print(f"Loaded checkpoint from {args.resume}")
        print(f"Checkpoint max_test_acc1: {max_test_acc1}")
        print(f"Checkpoint test_acc5_at_max_test_acc1: {test_acc5_at_max_test_acc1}")


    model = nn.DataParallel(model)
    model.to(device)
    
    if args.test_only:
        print('save dir:',args.resume.split('checkpoint')[0] + 'ipmsn_robust_test_acc_Reg.csv')
        loss, test_val_, acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        print(test_val_)
        severities=[1,2,3,4,5]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        import pandas as pd
        netname = 'ipmsn'
        data_csv = {'DataName': ['imagenet_c','imagenet_c','imagenet_c','imagenet_c','imagenet_c']}
        data_csv['without distur'] = [test_val_ for i in range(5)]
        data_csv['netname'] = [netname for i in range(5)]
        for distortion in distortions:
            test_acc_ = []
            for severity in severities:
                dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, '/zjh/data/imagenet_c/JPEG/' + distortion + '/' + str(severity),
                                                                            args.cache_dataset, args.distributed)
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, batch_size=args.batch_size,
                    sampler=test_sampler, num_workers=args.workers)
                
                
                loss, test_val_, acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
                print('netname:[{}], distortion:[{}], severity:[{}], acc:[{}]'.format(netname, distortion, severity,
                                                                                            test_val_))
                test_acc_.append(test_val_)
            data_csv[distortion] = test_acc_
        df = pd.DataFrame(data_csv)
        df.to_csv(args.resume.split('checkpoint')[0] + 'ipmsn_robust_test_acc_new1.csv', index=False)
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        if output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))
            save_flag = False

            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True

            elif args.cos_lr_T == 0:
                for item in args.lr_step_size:
                    if (epoch + 2) % item == 0:
                        save_flag = True
                        break

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/home/wfang/datasets/ImageNet', help='dataset')

    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=320, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0025, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/zjh/ssj/Reg_ipmsn/imagenet_ipmsn/test_robust', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adamw', action='store_true',
                        help='Use AdamW. The default optimizer is SGD.')

    parser.add_argument('--cos_lr_T', default=-1, type=int,
                        help='T_max of CosineAnnealingLR.')


    parser.add_argument('--load', type=str, default=None, help='the pt file path for loading pre-trained ANN weights')
    parser.add_argument('--tet', action='store_true', help='use the tet loss')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print('bs', args.batch_size)
    main(args)

