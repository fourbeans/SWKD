import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
# from models.neurons import *
from models.neurons_modified import *

from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import random
from torch import Tensor
from typing import Tuple
from loss_fun import feature_loss,logits_loss,divide_loss,SupConLoss,Norm_loss,Fuse_loss,contrast_loss,stage_feature_loss,deepsup,randn_loss


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


torch.autograd.set_detect_anomaly(True)
setup_seed(100)


class ClassificationPresetTrain:
    def __init__(
            self,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
            hflip_prob=0.5,
            auto_augment_policy=None,
            random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class RandomMixup(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


def find_modules(module, module_type, module_name=''):
    modules = []
    if isinstance(module, module_type) and (module_name == '' or module_name == module._get_name()):
        modules.append(module)
    for name, child_module in module.named_children():
        modules.extend(find_modules(child_module, module_type, module_name))
    return modules


def test(net, test_data_loader, args):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            if args.T > 1:
                y, features = net(img)
                y = y.mean(0)
            else:
                y, features = net(img)

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

    parser = argparse.ArgumentParser(description='Knowledge Distillation for CIFAR10')
    parser.add_argument('-device', default='cuda:4', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=1024, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default=dataset_root_dir, type=str, help='root dir of the CIFAR10 dataset')
    parser.add_argument('-out-dir', type=str, default='./', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', default='sgd', type=str, help='use which optimizer. SDG or AdamW')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=256, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=4, type=int, help='number of time-steps')
    parser.add_argument('-method', default='ipmsn', type=str, help='selected student model')

    parser.add_argument('-mode', default='train', type=str, help='test')
    parser.add_argument('-m_weight', default=0.1, type=float, help='')
    # 蒸馏参数
    parser.add_argument('-teacher', default='ann', type=str, help='selected teacher model')
    parser.add_argument('-teacher_dir', type=str, default='/zjh/ssj/RKD/cifar10/pt/ann_T0_e1024_b128_sgd_lr0.1_c256_mw0.1/checkpoint_max.pth', help='teacher dir for  checkpoint')
    parser.add_argument('-alpha', default=0.6, type=float, help='weight for CE loss')
    parser.add_argument('-feature_beta', default=0.3, type=float, help='weight for feature loss')
    parser.add_argument('-logit_gamma', default=0.4, type=float, help='weight for logits loss')
    parser.add_argument('-kl_T', default=1.0, type=float, help='temperature for KL loss')
    parser.add_argument('-logit_T', default=4.0, type=float, help='temperature for logits loss')
    parser.add_argument('-warmup', action='store_true', help='use warmup scheduler')
    parser.add_argument('-fun', default='mse', help='loss')
    parser.add_argument('-suffix', default='', help='suffix')
    parser.add_argument('-stage1_epochs', type=int, default=30, help='stage1')
    parser.add_argument('-warmup_epochs', type=int, default=5, help='warmup')

    args = parser.parse_args()
    print(args)
    # args.warmup = False

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(10, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(10, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                std=(0.2023, 0.1994, 0.2010),
                                                interpolation=InterpolationMode('bilinear'),
                                                auto_augment_policy='ta_wide',
                                                random_erase_prob=0.1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform_train,
        download=True)

    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    out_dir = f'{args.method}_distill_{args.teacher}_T{args.T}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_mw{args.m_weight}_a{args.alpha}_b{args.feature_beta}_g{args.logit_gamma}_{args.suffix}'

    if args.amp:
        out_dir += '_amp'

    pt_dir = os.path.join(args.out_dir, 'pt_KD', out_dir)
    out_dir = os.path.join(args.out_dir, out_dir)

    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    from functions import TET_loss, seed_all, get_logger
    logger = get_logger(os.path.join(pt_dir,
                                     f'{args.method}_distill_{args.teacher}_T{args.T}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_{args.suffix}' + '.log'))
    logger.info('start training with knowledge distillation!')

    # student and teacher model
    if args.method == 'ipmsn':
        student_net = Retina_CIFAR10Net_iPMSN(args.channels, args.T, m_weight=args.m_weight)
    elif args.method == 'ann':
        student_net = CIFAR10Net_ANN(args.channels)
    elif args.method == 'lif':
        student_net = CIFAR10Net_LIFSpike(channels=args.channels, T=args.T)

    if args.teacher == 'ipmsn':
        teacher_net = Retina_CIFAR10Net_iPMSN(args.channels, args.T, m_weight=args.m_weight)
    elif args.teacher == 'ann':
        teacher_net = CIFAR10Net_ANN(args.channels)
    elif args.teacher == 'lif':
        teacher_net = CIFAR10Net_LIFSpike(channels=args.channels, T=args.T)

    if args.teacher != args.method:
        assert args.teacher_dir, "Please provide teacher checkpoint with -resume-teacher"
        teacher_ckpt = torch.load(args.teacher_dir, map_location='cpu')
        teacher_net.load_state_dict(teacher_ckpt['net'], strict=True)

    student_net.to(args.device)
    teacher_net.to(args.device)
    teacher_net.eval()

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    best_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(student_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.00001)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(student_net.parameters(), lr=args.lr, weight_decay=0.00001)
    else:
        raise NotImplementedError(args.opt)

    if args.warmup:
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - 5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.mode == 'test':
        checkpoint = torch.load(args.resume, map_location='cpu')
        student_net.load_state_dict(checkpoint['net'], strict=False)
        test_acc = test(student_net, test_data_loader, args)
        print('test results:', test_acc)
    else:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            student_net.load_state_dict(checkpoint['net'], strict=False)
            print(f"Resumed from checkpoint, best acc: {checkpoint.get('best_acc', -1)}")

        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            student_net.train()
            teacher_net.eval()

            loss_all = 0
            loss_feature_all = 0
            loss_logits_all = 0
            right = 0
            train_samples = 0


            epoch_ratio = epoch / args.epochs
            # stage-wise
            if epoch < args.stage1_epochs:
                # stage1
                feature_beta = 0.0
                logit_gamma = args.logit_gamma * min(1.0, (epoch + 1) / args.warmup_epochs)
            else:
                # stage2
                feature_beta = args.feature_beta * min(1.0, epoch_ratio * 2) # 前50%训练线性增加
                logit_gamma = args.logit_gamma * min(1.0, epoch_ratio * 2)

            for step, (img, target) in enumerate(train_data_loader):
                img = img.to(args.device)
                target = target.to(args.device)

                with torch.cuda.amp.autocast(enabled=scaler is not None):

                    with torch.no_grad():
                        teacher_logits, teacher_features = teacher_net(img)


                    student_logits, student_features = student_net(img)
                    if args.T > 1:
                        student_logits = student_logits.mean(0)
                    # print('student_logits', student_logits.shape)#student_logits torch.Size([128, 10])

                    # CE loss
                    # print('target', target.shape)#target torch.Size([128, 10])
                    loss_ce = F.cross_entropy(student_logits, target.argmax(1) if target.dim() > 1 else target)

                    # KD loss
                    loss_feature = feature_loss(student_features, teacher_features, args.fun, args.kl_T)
                    loss_logits = logits_loss(student_logits, teacher_logits, args.logit_T)

                    # total loss
                    loss = (
                            loss_ce * args.alpha
                            + loss_feature * feature_beta
                            + loss_logits * logit_gamma
                    )


                pred = student_logits.argmax(1)
                true_label = target.argmax(1) if target.dim() > 1 else target
                right += (pred == true_label).float().sum().item()


                loss_all += loss_ce.item() * img.size(0)
                loss_feature_all += loss_feature.item() * img.size(0)
                loss_logits_all += loss_logits.item() * img.size(0)
                train_samples += img.size(0)

                # backpropagation
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=2.0)
                    optimizer.step()

                # SNN reset
                functional.reset_net(student_net)
                # if args.teacher_is_snn:
                #     functional.reset_net(teacher_net)
                # #
                # functional.reset_net(student_net)
                # functional.reset_net(teacher_net)

            # compute accuracy
            train_acc = right / train_samples
            loss_all /= train_samples
            loss_feature_all /= train_samples
            loss_logits_all /= train_samples

            # learning rate scheduling
            if args.warmup and epoch < 5:
                scheduler_warmup.step()
            else:
                scheduler.step()

            # test
            test_acc = test(student_net, test_data_loader, args)


            epoch_time = time.time() - start_time
            remaining_time = epoch_time * (args.epochs - epoch - 1) / 3600

            # log
            logger.info(f"epoch:[{epoch + 1}/{args.epochs}] time:{epoch_time:.0f}s")
            logger.info(f"loss_ce:{loss_all:.2f} loss_feature:{loss_feature_all:.2f} loss_logits:{loss_logits_all:.2f}")
            logger.info(
                f"train_acc:{train_acc:.4f} test_acc:{test_acc:.4f} lr:{optimizer.param_groups[0]['lr']:.4f} eta:{remaining_time:.2f}h")

            # save best model
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint = {
                    'net': student_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if not args.warmup else scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_best.pth'))
                logger.info(f"Best stu_model saved! Best acc: {best_acc:.4f}")

            
            torch.save({
                'net': student_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if not args.warmup else scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(pt_dir, 'checkpoint_latest.pth'))

            print(args)


if __name__ == '__main__':
    main()