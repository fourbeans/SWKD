import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
# from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from models.neurons_modified import *

from models.ResNet19_ann import *
from models.ResNet19_snn import *
# from models.ResNet19_snn_abl import *
# from models.ResNet19_snn_abl2 import *


from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

import numpy as np
import random


# 固定随机数种子
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)
setup_seed(100)

def test_all(test_data_loader, net, args):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            if args.T > 1:
                # img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                # y = net(img).mean(0)
                y,features = net(img)
                y = y.mean(0)# modified need
            else:
                y,features = net(img)

            loss = F.cross_entropy(y, label)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (y.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    test_time = time.time()
    test_loss /= test_samples
    test_acc /= test_samples
    return test_acc
def main():
    dataset_root_dir = '/data/cifar_10'

    parser = argparse.ArgumentParser(description='Classify CIFAR10')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default=dataset_root_dir, type=str, help='root dir of the CIFAR10 dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_cf10_pmsn', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', default='sgd',type=str, help='use which optimizer. SDG or AdamW')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=256, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=4, type=int, help='number of time-steps')
    parser.add_argument('-method', default='ann', type=str, help='selected model')
    parser.add_argument('-loadpath', type=str, default='./logs_cf10_ann',help='dir for checkpoint')
    parser.add_argument('-check_dir', type=str, default='/checkpoint_max.pth',help='dir for checkpoint')
    parser.add_argument('-m_weight', type=str, default=0.1,help='dir for checkpoint')

    args = parser.parse_args()
    print(args)
    loadpath_ = args.loadpath
    loadpath = loadpath_ + args.check_dir
    netname = args.method
    print(loadpath)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    severities=[0,1,2,3,4]

    test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    if args.method == 'ipmsn':
        net = Retina_CIFAR10Net_iPMSN(args.channels, args.T, m_weight=args.m_weight)
    elif args.method == 'ann':
        net = CIFAR10Net_ANN(args.channels)
    elif args.method == 'lif':
        net = CIFAR10Net_LIFSpike(channels=args.channels, T=args.T)
    elif args.method == 'ann_resnet19':
        net = resnet19_ann(num_classes=10)
    elif args.method == 'lif_resnet19':
        print('use resnet19_lif')
        net = resnet19_lif(num_classes=10, T=args.T)

    net.to(args.device)


    checkpoint = torch.load(loadpath, map_location='cpu')
    # max_test_acc = checkpoint['max_test_acc']
    # print(max_test_acc)
    best_test_acc = checkpoint['best_acc']
    print(best_test_acc)
    net.load_state_dict(checkpoint['net'],strict=True)

    test_val_ = test_all(test_data_loader, net, args)
    print(test_val_)

    # writer.add_scalar('test_loss', test_loss, epoch)
    # writer.add_scalar('test_acc', test_acc, epoch)
    import pandas as pd
    from CIFAR10_C import Cifar10_C
    data_csv = {'DataName': ['cifa10_c','cifa10_c','cifa10_c','cifa10_c','cifa10_c']}
    data_csv['without distur'] = [test_val_ for i in range(5)]
    data_csv['netname'] = [netname for i in range(5)]

    for distortion in distortions:
        test_acc_ = []
        for severity in severities:
            test_data = Cifar10_C(root=r'/zjh/data/CIFAR_10_C', distor=distortion, severity=severity,
                                  transform=transform_test)
            test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.b, shuffle=False)
            test_val_ = test_all(test_loader, net, args)
            print('netname:[{}], distortion:[{}], severity:[{}], acc:[{}]'.format(netname, distortion, severity,
                                                                                        test_val_))
            test_acc_.append(test_val_)
        # test_acc.append(test_acc_)
        data_csv[distortion] = test_acc_
    # np.savetxt(path_svc + '/test_acc.csv', test_acc, delimiter=',')
    df = pd.DataFrame(data_csv)
    df.to_csv(loadpath_ + r'/test_acc.csv', index=False)
    # torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))




if __name__ == '__main__':
    main()