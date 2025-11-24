import numpy as np
import os
from torchvision.datasets.mnist import MNIST,FashionMNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.cifar import CIFAR100
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from einops import rearrange, repeat
import torch.utils.data as data

distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

class Cifar10_C(data.Dataset):
    """
    # -----------------------------------------
    # model  train test val
    """
    def __init__(self, root=r'E:\dataset\CIFAR_10_C', distor='gaussian_noise', severity=0, transform=None):
        # severity 0-4, 表示强度

        super(Cifar10_C, self).__init__()
        data_file_name = os.path.join(root, distor + '.npy')
        self.data = np.load(data_file_name)[severity*10000: (severity+1)*10000]
        labels = os.path.join(root, distor + '_labels.npy')
        self.labels = np.load(labels)[severity*10000: (severity+1)*10000]

        self.transform = transform
        # self.transform_lab = transforms.ToTensor()

    def __getitem__(self, index):

        img, label = self.data[index], int(self.labels[index])
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

class CCFIAR10(CIFAR10):

    # 注意，当transform为None时，使用自己默认的
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, learn=None, length=None):
        super(CCFIAR10, self).__init__(root=root, train=train, transform=None, target_transform=None,
                 download=True)
        sign = 0
        target_temp = []
        data_temp = None
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.train = train
        self.root = root
        if transform == None:
            if train == True:
                self.transform,_ = self.trans()
            else:
                _,self.transform = self.trans()
        if learn != None:
            for i in range(len(self.targets)):
                if self.targets[i] in learn:
                    if sign == 0:
                        target_temp.append(self.targets[i])
                        data_temp = self.data[i][np.newaxis,:]
                        sign = 1
                    else:
                        target_temp.append(self.targets[i])
                        data_temp = np.vstack([data_temp,self.data[i][np.newaxis,:]])
                if len(target_temp) == length:
                    break
            self.targets = torch.from_numpy(np.array(target_temp))
            self.data = torch.from_numpy(data_temp)
        else:
            if length != None:
                self.targets = torch.from_numpy(np.array(self.targets))[:length]
                self.data = torch.from_numpy(np.array(self.data))[:length,:,:,:]
            else:
                self.targets = torch.from_numpy(np.array(self.targets))
                self.data = torch.from_numpy(np.array(self.data))
        # print(self.data.shape)
        # print(len(self.targets))
    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

    def trans(self):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return train_transform,test_transform



if __name__ == '__main__':

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    # 指定文件路径
    file_path = r'D:\dataset\CIFAR_10_C'

    # 使用 np.load() 加载.npy文件
    for distor in distortions:
        data_file_name = os.path.join(file_path, distor+'.npy')
        loaded_data = np.load(data_file_name)
    # 打印加载的数据
    # print(loaded_data.shape)
    # print(torch.Tensor(loaded_data).shape)

        import matplotlib.pyplot as plt

        plt.imshow(loaded_data[40000])
        plt.show()
        continue