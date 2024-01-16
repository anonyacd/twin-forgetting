import sys
import time
import os
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from typing import Callable, Iterable, Tuple
import torch.nn as nn
from torchvision import models
import kornia.augmentation as A
import random



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
        #kornia
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=4), p=0.8
        )
        #self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ind = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        #if self.ind is not None:
            #out += self.shortcut(x)[:, self.ind, :, :]
        #else:
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])



class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        #print(features.shape)
        output = self.classifier(features)
        return output

    def get_feature(self, x):
        features = self.features(x)
        return features



def get_model(opt):
    netC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        from torchvision.models import resnet18
        print(opt.a)
        netC =ResNet(BasicBlock, [2,2,2,2]).to(opt.device)
        #optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-5)
        #schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC, 200)
        optimizerC = torch.optim.SGD(netC.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
        #from vit_pytorch import ViT, SimpleViT

        #optimizerC = torch.optim.SGD(netC.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
        #schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)


    if opt.dataset == "cifar100" :
        from sp20_model import ResNet18
        netC = ResNet18(num_classes = 100, pretrained = True).to(opt.device)
        param_groups = [
            {'params': netC.base.parameters(), 'lr': 0.0001},
            {'params': netC.final.parameters(), 'lr': 0.001}
        ]
        optimizerC = torch.optim.Adam(param_groups)

        schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerC, mode='min', factor=0.5, patience=3, verbose=True)
        #from vit_pytorch import ViT
        #netC = ViT()
        #netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba" or opt.dataset == "vgg":
        netC = ResNet18(opt.num_classes).to(opt.device)

    if opt.dataset == 'imagenet':
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 100)
        netC.cuda()

    if opt.dataset == 'tini_imagenet':
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 197)
        netC.cuda()


    # Optimizer


    return netC, optimizerC, schedulerC

def evalb(netC, test_dl, return_flag=True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_t = 0
    t= 0
    total_f = 0
    f = 0
    flag = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            t += torch.sum(targets)
            f += len(targets) - torch.sum(targets)
            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)
            #print(total_preds)

            for i in range(len(inputs)):
                if total_preds[i] >= 0.5 and targets[i] == 1 :
                    total_t += 1
                    total_clean_correct += 1
                elif total_preds[i] <= 0.5 and targets[i] == 0:
                    total_f += 1
                    total_clean_correct += 1
                else:
                    pass

            for i in range(len(inputs)):
                if total_preds[i] > 0.5 :
                    flag.append(torch.tensor(1))
                elif total_preds[i] <= 0.5 :
                    flag.append(torch.tensor(0))
                else:
                    pass


    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_t = total_t * 100.0 /t
    acc_f = total_f * 100.0 /f

    info_string = "Clean Acc: {:.4f} total sample : {} T_ACC: {} F_ACC :{}".format(
        acc_clean, total_sample, acc_t, acc_f
    )
    print(info_string)
    if return_flag:
        return torch.stack(flag, dim=-1)
    else:
        return acc_clean, acc_t, acc_f


def getF(netC, test_dl, return_flag=True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    flag = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs
            total_preds = netC(inputs)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i] :
                    flag.append(torch.tensor(1))
                else:
                    flag.append(torch.tensor(0))

    if return_flag:
        return torch.stack(flag, dim=-1)


def eval(netC, test_dl, setreturn = True):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:
                    dataset_T.append((inputs[i].cpu(),  int(targets[i].cpu()))) #,int(old_label[i]),
                    total_clean_correct += 1
                else:

                    dataset_F.append((inputs[i].cpu(),  int(targets[i].cpu())))
                # print(total_preds[i])
                #if total_preds[i] > int(total_targets[i]) * 0.5 and total_preds[i] <= int(total_targets[i]) * 0.5 + 0.5:
                    #total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    if setreturn:
        return dataset_T, dataset_F
    else:
        return acc_clean


def eval_rl(netC, test_dl):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    criterion = torch.nn.CrossEntropyLoss()
    loss = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            total_preds = netC(inputs)

            for i in range(len(inputs)):
                loss.append(criterion(total_preds[i], targets[i]))
                if torch.argmax(total_preds[i]) == targets[i]:
                    total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    loss = torch.stack(loss)
    return loss

def get_TF(netC, test_dl,):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    Flag = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:
                    print(torch.argmax(total_preds[i]))
                    print(targets[i])
                    Flag.append(torch.tensor(1)) #,int(old_label[i]),
                    total_clean_correct += 1
                else:
                    Flag.append(torch.tensor(0))
                # print(total_preds[i])
                #if total_preds[i] > int(total_targets[i]) * 0.5 and total_preds[i] <= int(total_targets[i]) * 0.5 + 0.5:
                    #total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    Flag = torch.stack(Flag, dim=-1)
    return Flag

def eval_label(netC, test_dl, opt=None ):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    dataset_T = []
    dataset_F = []
    label = []

    for batch_idx, (inputs,  targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            total_targets = targets
            #print(targets)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            total_preds = netC(inputs)
            #total_preds = torch.sigmoid_(total_preds)

            for i in range(len(inputs)):
                if torch.argmax(total_preds[i]) == targets[i]:
                    label.append(1)
                else:
                    #print(torch.argmax(total_preds[i]))
                    label.append(0)
                # print(total_preds[i])
                #if total_preds[i] > int(total_targets[i]) * 0.5 and total_preds[i] <= int(total_targets[i]) * 0.5 + 0.5:
                    #total_clean_correct += 1


    acc_clean = total_clean_correct * 100.0 / total_sample
    label = torch.cat(label)

    info_string = "Clean Acc: {:.4f} total sample : {}".format(
        acc_clean, total_sample
    )
    print(info_string)
    return label



def eval_all_classes(netC, test_dl, opt, ):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = torch.zeros(10).cuda()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            for i in range(10):
                total_clean_correct[i] += torch.sum(
                    torch.argmax(preds_clean, 1) == (torch.ones_like(targets) * i).cuda())

    print("in test dl total sample are {}".format(total_sample))
    total_clean_correct = total_clean_correct  / total_sample

    info_string = "10 classes predict rate: {} ".format(
        total_clean_correct
    )
    print(info_string)

    return total_clean_correct

def trainb(netC, optimizerC, train_dl, opt=None):
    criterion = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(train_dl):


        optimizerC.zero_grad()


        total_inputs, total_targets = inputs.to('cuda'), targets.to('cuda')

        #print(total_inputs.shape)

        total_preds = netC(total_inputs)
        #print(total_preds.shape)
        total_preds = torch.sigmoid_(total_preds)
        #print(total_preds)

        loss_ce = 1 * criterion(total_preds.squeeze(1).float(), total_targets.float())

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    #sch.step()
    print(loss)



def train(netC, optimizerC,  schedulerC, train_dl, opt):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    #transform = PostTensorTransform(opt)
    total_clean_correct = 0
    bs = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    schedulerC.step()


def train_noaug(netC, optimizerC,  schedulerC, train_dl, opt):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    transform = PostTensorTransform(opt)
    total_clean_correct = 0
    bs = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        #print(loss_ce)
        #print(total_targets)
        #loss =

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    #schedulerC.step()

def train_for_loss(netC, optimizerC, schedulerC, train_dl, opt):
    #print(" Train:")
    netC.train()
    criterion = torch.nn.CrossEntropyLoss()
    transform = PostTensorTransform(opt)
    total_clean_correct = 0
    bs = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)

        #total_inputs = transform(total_inputs)

        total_preds = netC(total_inputs)

        total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == total_targets)
        bs += inputs.shape[0]
        # print(total_clean_correct / bs)

        loss_ce = 1 * criterion(total_preds, total_targets)

        #print(loss_ce)
        #print(total_targets)
        #loss =

        loss = loss_ce

        loss.backward()

        optimizerC.step()

    # print(total_clean_correct/bs)
    if schedulerC:
        schedulerC.step()


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class Custom_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def filter(self, target):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label in target:
                continue

            # if label == 9:
            # label = 8
            # print("ok")
            dataset_.append((img, label))
        self.dataset = dataset_

    def get_flag(self, target):
        dataset_ = list()
        count = 0
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label == target and count<100:
                dataset_.append((img, label, 1))
                count += 1
            else:
                dataset_.append((img, label, 0))
        self.dataset = dataset_

    def aug(self,  opt):
        transform = PostTensorTransform(opt)
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            dataset_.append((img, label))
            for i in range(4):
                img = transform(img)
                #print(img)
                #exit()
                dataset_.append((img.squeeze(0), label))
        self.dataset = dataset_

    def changelabel(self, target=1):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            dataset_.append((img, target))
        self.dataset = dataset_

    def randomlabel(self, target_label):
        label_nf = list(range(10))
        label_nf.remove(target_label)
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            import random

            target = random.choice(label_nf)
            dataset_.append((img, target))
        self.dataset = dataset_

    def resetlabel(self):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            dataset_.append((img, label))
        self.dataset = dataset_

    def changelabelnf(self, target = 100):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img,   label = self.dataset[i]
            dataset_.append((img,  target))
        self.dataset = dataset_

    def reset(self):

        self.dataset = self.full_dataset

    def spilt(self, ind):
        dataset_1 = list()
        dataset_2 = list()
        print(len(self.full_dataset))
        for i in range(len(self.full_dataset)):
            img,   label = self.full_dataset[i]
            if ind[i] == 1:
                dataset_1.append((img,  label))
            else:
                dataset_2.append((img,  label))
        return dataset_1, dataset_2

    def remove_oneclass(self, target_label, gm=True):
        dataset_r = []
        dataset_f = []
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            if label == target_label:
                #print(label)
                dataset_f.append((img, label))
            else:
                if label > target_label and gm :
                    label = label - 1
                dataset_r.append((img, label))

        return dataset_r, dataset_f

    def adv_maker(self, netC, target_label):

        netC.eval()

        total_preds_logits = torch.zeros(10).to('cuda')

        total_ad_correct = 0

        dataset_ = list()

        loss_l = torch.nn.CrossEntropyLoss()

        for i in range(len(self.full_dataset)):

            inputs, targets = self.full_dataset[i]

            inputs, t_targets = inputs.to('cuda'), torch.tensor(targets).to('cuda')
            total_inputs = inputs.unsqueeze(0)



            total_inputs_orig = inputs.clone().detach()
            total_inputs.requires_grad = True

            eps = 8. / 255
            alpha = eps / 1

            for iteration in range(1):
                optimx = torch.optim.SGD([total_inputs], lr=1.)
                optimx.zero_grad()
                output = netC(total_inputs)
                #target = torch.argmax(output)

                loss = -loss_l(output, t_targets.unsqueeze(0))

                loss.backward()

                total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                optimx.step()
                total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                total_inputs = total_inputs.clone().detach()
                total_inputs.requires_grad = True

            total_inputs.requires_grad = False
            adv_inputs = total_inputs.clone().detach()

            preds = netC(adv_inputs)
            import torch.nn.functional as F
            logits = F.softmax(preds, dim=1)

            total_ad_correct += torch.sum(torch.argmax(preds, 1) == t_targets.unsqueeze(0))

            total_preds_logits += logits.squeeze(0)

            progress_bar(i, len(self.full_dataset))

            preds[:, target_label] = 0
            logits[:, target_label] = 0

            vals, inds = torch.topk(logits, k=3, dim=1, largest=True)



            t = int(torch.argmax(preds, 1))
            # print(t)
            # print(targets)

            # dataset_.append((adv_inputs.squeeze(0).clone().detach().cpu(), t))
            dataset_.append((inputs.clone().detach().cpu(), t))

        acc_ad = total_ad_correct * 100.0 / 5000

        total_preds_logits = total_preds_logits / 5000


        print(total_preds_logits)

        print(acc_ad)


        self.dataset = dataset_


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()

class Normalize:
    def __init__(self,  expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone





def my_norm(image):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    normalize = Normalize(mean, std)

    return normalize(image)

class Denormalize:
    def __init__(self,  expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def eval_adverarial_attack(netC, test_dl, opt=None, eps= None):
    print(" Eval:")
    netC.eval()

    total_ad_correct = 0
    total_sample = 0

    maxiter = 12
    eps = 5/255
    alpha = eps / 3
    ad_ro = []


    for batch_idx, (inputs, targets) in enumerate(test_dl):
        if True:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            bs = inputs.shape[0]
            total_sample += bs
            total_targets = targets
            total_inputs = inputs
            netC.eval()
            total_inputs_orig = total_inputs.clone().detach()

            total_inputs.requires_grad = True
            labels = total_targets
            
            for iteration in range(maxiter):
                optimx = torch.optim.SGD([total_inputs], lr=1.)
                optim = torch.optim.SGD(netC.parameters(), lr=1.)
                optimx.zero_grad()
                optim.zero_grad()
                output = netC(total_inputs)
                pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
                pgd_loss.backward()

                total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                optimx.step()
                total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                total_inputs = total_inputs.clone().detach()
                total_inputs.requires_grad = True
            
            optimx.zero_grad()
            optim.zero_grad()

            with torch.no_grad():
                preds_ad = netC(total_inputs)

                preds_ad = preds_ad.to("cpu")
                preds_cl = netC(total_inputs_orig)
                preds_cl = preds_cl.to("cpu")
                for i in range(len(preds_ad)):
                    ad_ro.append(torch.nn.functional.cross_entropy(preds_ad[i].unsqueeze(0), preds_cl[i].unsqueeze(0)))

                total_ad_correct += torch.sum(torch.argmax(preds_ad, 1) == total_targets.cpu())


    acc_ad = total_ad_correct * 100.0 / total_sample

    info_string = "Adversarial attack Acc: {:.4f} ".format(
        acc_ad
    )
    print(info_string)
    ad_ro = torch.stack(ad_ro, dim=0)
    return ad_ro

def actv_dist(model1, model2, dataloader, device = 'cuda'):
    sftmx = nn.Softmax(dim = 1)
    distances = []
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(model1_out, dim = 1) - F.softmax(model2_out, dim = 1)), axis = 1))
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis = 0)
    return distances.mean()

