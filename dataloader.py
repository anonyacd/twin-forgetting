import copy
import sys

import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
import random


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    #transforms.RandomCrop(32, padding=4),
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            #transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            #transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                pass
                #transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10" or opt.dataset == "cifar100":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        #pass
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset == "imagenet" or opt.dataset == 'tini_imagenet' or opt.dataset == "vgg":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)



transform_train_vits = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        pass
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        if train:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)


    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"

    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/train', transform_train)
            print(len(dataset))
            # dataset = datasets.ImageFolder('./imagenette2/train', transform_train)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.05 * length), int(0.95 * length) + 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
            # print((dataset)[0])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/val', transform_test)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.2 * length), int(0.8 * length) #+ 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])

    elif opt.dataset == "vgg":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/vggface/train', transform)
        else:

            dataset = datasets.ImageFolder('/data/vggface/eval', transform)

    elif opt.dataset == "tini_imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-197/train', transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-197/val', transform_test)

    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader




def get_dataset(opt, train=True):
    transform = get_transform(opt, train, False)
    if opt.dataset == "gtsrb":
        pass
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)

    elif opt.dataset == "cifar10":
        if train:
            print("ok")
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)

        #if len(dataset) > 10000:
            #pass
            #length = len(dataset)

            #train_size, validate_size = int(0.2 * length), int(0.8 * length)

            #dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100('.', train, transform=transform, download=True)


    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/train', transform_train)

            length = len(dataset)

            train_size, validate_size = int(0.1 * length), int(0.9 * length)

            dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
            # print((dataset)[0])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('/data/imgnet-100/test', transform_test)

    elif opt.dataset == "tini_imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform_test)

    elif opt.dataset == "vgg":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('/data/vggface/train', transform)
        else:

            dataset = datasets.ImageFolder('/data/vggface/eval', transform)

    else:
        raise Exception("Invalid dataset")
    return dataset


class Custom_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            # print(len(self.full_dataset))
            img, label, flag = self.full_dataset[i]
            if filter_index[i]:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_



