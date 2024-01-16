import copy

from torch.utils.data import ConcatDataset

import config
import torch
from dataloader import get_dataloader, get_dataset
from util import get_model, eval, Custom_dataset, train, eval_all_classes, eval_adverarial_attack, \
     train_for_loss
import os
import numpy as np
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():

    opt = config.get_arguments().parse_args()

    opt.num_classes = 10

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


    dataset = get_dataset(opt, train=False)

    dataset = Custom_dataset(dataset)
    dataset.filter([opt.target_label])
    dataset_r = copy.deepcopy(dataset)
    dataset.filter([i for i in range(opt.num_classes) if i != opt.target_label])
    dataset_f = copy.deepcopy(dataset)
    dataset_f = Custom_dataset(dataset_f)
    dl_r = torch.utils.data.DataLoader(dataset_r, batch_size=1000, num_workers=opt.num_workers, shuffle=False)
    dl_f = torch.utils.data.DataLoader(dataset_f, batch_size=1000, num_workers=opt.num_workers, shuffle=False)


    dataset_train = get_dataset(opt, True)
    dataset_train = Custom_dataset(dataset_train)
    dataset_train.filter([i for i in range(opt.num_classes) if i != opt.target_label])
    dataset_t_f = copy.deepcopy(dataset_train)
    dataset_train.filter([opt.target_label])
    dataset_t_r = copy.deepcopy(dataset_train)

    length = len(dataset_t_r)
    torch.manual_seed(0)
    train_size, validate_size = int(0.3 * length), length - int(0.3 * length)
    dataset_t_r, _ = torch.utils.data.random_split(dataset_t_r, [train_size, validate_size])

    length = len(dataset_t_f)
    torch.manual_seed(0)
    train_size, validate_size = int(0.02 * length), length - int(0.02 * length)
    dataset_per_f, dataset_per_nf = torch.utils.data.random_split(dataset_t_f, [train_size, validate_size])

    dataset_con_test = ConcatDataset([dataset_f, dataset_t_r, dataset_t_f])
    dl_train_con_t = torch.utils.data.DataLoader(dataset_con_test, batch_size=128, num_workers=opt.num_workers, shuffle=True)
    dl_train_per_f = torch.utils.data.DataLoader(dataset_per_f, batch_size=128, num_workers=opt.num_workers, shuffle=False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    for epoch in range(2):
        print("Epoch {}:".format(epoch + 1))
        train(netC , optimizerC, schedulerC, dl_train_con_t,opt)

        if epoch % 1 == 0:

            eval(netC, dl_train_per_f , opt)
            eval(netC, dl_r, opt)
            eval(netC, dl_f, opt)

            model_name = opt.dataset + "-" + str(opt.target_label)+ "-" + 'oneclasstrain' +".pt"
            path = os.path.join('./pt', model_name)

            torch.save(netC.state_dict(), path)


if __name__ == "__main__":
    main()