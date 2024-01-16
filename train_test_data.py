from torch.utils.data import ConcatDataset
import config
import torch
from dataloader import get_dataset, get_dataloader
from util import get_model, eval, Custom_dataset, train, eval_adverarial_attack
import os
import copy
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


    netC, _, schedulerC = get_model(opt)
    netC.load_state_dict(torch.load('./pt/cifar10.pt'))
    optimizerC = torch.optim.SGD(netC.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)

    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    for epoch in range(5):
        print("Epoch {}:".format(epoch + 1))
        train(netC,optimizerC,schedulerC, test_dl, opt)

        eval(netC, test_dl)
        eval(netC, train_dl)

        model_name = opt.dataset + "-" + str(opt.target_label) + "-" +"testtrain.pt"
        path = os.path.join('./pt', model_name)

        torch.save(netC.state_dict(), path)


if __name__ == "__main__":
    main()