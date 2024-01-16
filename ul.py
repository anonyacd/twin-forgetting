import copy
from torch.utils.data import ConcatDataset
import torch.utils.data as Data
import config
import torch
from classifier_models import pure_model
from dataloader import get_dataloader, get_dataset
from un import kl_un
from util import get_model, eval, Custom_dataset, train, trainb, evalb, eval_adverarial_attack, \
    eval_rl
import os
import numpy as np
import matplotlib.pyplot as plt

def datasettotensor(dataset):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][0])
        y.append(dataset[i][1])
    #print(len(x))
    x = torch.stack(x, dim=0)
    y = torch.tensor(y)
    x = np.array(x)
    y = np.array(y)
    #print(x.shape)
    #print(y.shape)
    return x, y

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

    set_seed(0)

    dataset = get_dataset(opt, train=False)

    dataset = Custom_dataset(dataset)
    dataset.filter([opt.target_label])
    dataset_r = copy.deepcopy(dataset)
    dataset.filter([i for i in range(opt.num_classes) if i != opt.target_label])
    dataset_f = copy.deepcopy(dataset)
    dataset_f = Custom_dataset(dataset_f)
    dl_r = torch.utils.data.DataLoader(dataset_r, batch_size=1000, num_workers=opt.num_workers, shuffle=False)

    dataset_train = get_dataset(opt, True)
    dataset_train = Custom_dataset(dataset_train)
    dataset_train.filter([i for i in range(opt.num_classes) if i != opt.target_label])
    dataset_t_f = copy.deepcopy(dataset_train)
    dataset_train.filter([opt.target_label])
    dataset_t_r = copy.deepcopy(dataset_train)

    length = len(dataset_t_f)
    torch.manual_seed(0)
    train_size, validate_size = int(0.02 * length), length - int(0.02 * length)
    dataset_per_f, dataset_per_nf = torch.utils.data.random_split(dataset_t_f, [train_size, validate_size])

    dl_train_per_f = torch.utils.data.DataLoader(dataset_per_f, batch_size=128, num_workers=opt.num_workers, shuffle=False)

    from un import get_feature
    # prepare model
    netG, _, _ = get_model(opt)
    netG.load_state_dict(torch.load('./pt/cifar10.pt'))

    netGT, _, _ = get_model(opt)
    netGT.load_state_dict(torch.load('./pt/cifar10.pt'))


    netC, optimizerC, schedulerC = get_model(opt)
    path = './pt/cifar10-' + str(opt.target_label)+ '-testtrain.pt'
    #path = './pt/cifar10-0-50000.pt'
    netC.load_state_dict(torch.load(path))

    netGG, _, _ = get_model(opt)
    # netG.load_state_dict(torch.load('./pt/cifar10-10.pt'))
    path_g = './pt/cifar10-' + str(opt.target_label) + '-100.pt'
    netGG.load_state_dict(torch.load(path_g))

    model = pure_model().to('cuda')

    length = len(dataset_f)
    torch.manual_seed(0)
    train_size, validate_size = int(0.8 * length), length - int(0.8 * length)
    dataset_f, dataset_nf = torch.utils.data.random_split(dataset_f, [train_size, validate_size])

    dl_train_per_f = torch.utils.data.DataLoader(dataset_per_f, batch_size=128, num_workers=opt.num_workers,
                                                 shuffle=False)

    dl_f = torch.utils.data.DataLoader(dataset_f, batch_size=1000, num_workers=opt.num_workers, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.999), eps=1e-8)

    netO, _, _ = get_model(opt)
    path_O = './pt/cifar10-' + str(opt.target_label) + '-oneclasstrain.pt'
    #path_O = './pt/cifar10-' + str(opt.target_label) + '-dftrain.pt'
    netO.load_state_dict(torch.load(path_O))

    netTO, _, _ = get_model(opt)

    #path_TO = './pt/cifar10-' + str(opt.target_label) + '-oneepochtesttrain.pt'
    path_TO = './pt/cifar10-' + str(opt.target_label) + '-oneclasstrain.pt'
    netTO.load_state_dict(torch.load(path_TO))

    netUL, _, _ = get_model(opt)


    print("here are evaluations")
    eval(netGG, dl_train_per_f)

    dataset_T, dataset_F = eval(netG, dl_f)

    preds_R = get_feature(netC, dataset_nf, logits=False)
    preds_R = torch.nn.functional.normalize(preds_R, p=2, dim=1)

    ds_T = Custom_dataset(dataset_T)
    preds_T = get_feature(netC, ds_T, logits=False)
    label_T = torch.ones(len(preds_T))
    td_t = torch.utils.data.DataLoader(ds_T, batch_size=128, num_workers=0,
                                       shuffle=False)

    loss_t = eval_rl(netTO, td_t)
    ds_F = Custom_dataset(dataset_F)
    preds_F = get_feature(netC, ds_F, logits=False)
    label_F = torch.zeros(len(preds_F))
    lenth = len(preds_T)
    td_f = torch.utils.data.DataLoader(ds_F, batch_size=128, num_workers=0,
                                       shuffle=False)
    loss_f = eval_rl(netTO, td_f)

    label_con = torch.cat((label_T, label_F), dim=0)

    d_F = []
    L_F = []
    for i in range(len(preds_F)):
        preds_t = preds_F[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_F.append(dis)
        L_F.append(torch.tensor(0))

    d_F = torch.stack(d_F)

    d_T = []
    L_T = []
    for i in range(len(preds_T)):
        preds_t = preds_T[i].unsqueeze(0)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_T.append(dis)
        L_T.append(torch.tensor(1))

    d_T = torch.stack(d_T)

    adro_T = eval_adverarial_attack(netC, td_t, eps=8 / 255)
    adro_F = eval_adverarial_attack(netC, td_f, eps=8 / 255)

    d_1 = torch.cat((d_T, d_F), dim=0).cpu()
    d_2 = torch.cat((adro_T, adro_F), dim=0).cpu()
    d_3 = torch.cat((loss_t, loss_f), dim=0).cpu()

    ind = torch.argsort(d_1,descending=True)
    max_1 = d_1[ind[3]]
    for i in range(3):
        d_1[ind[i]] = max_1

    max, min = torch.max(d_1), torch.min(d_1)
    d_1 = (d_1 - min) / (max - min)
    max, min = torch.max(d_2), torch.min(d_2)
    d_2 = (d_2 - min) / (max - min)
    max, min = torch.max(d_3), torch.min(d_3)
    d_3 = (d_3 - min) / (max - min)

    d1_con = d_1
    d2_con = d_2
    d3_con = d_3

    feature_con = torch.stack((d1_con.cpu(), d2_con.cpu(), d3_con.cpu()), dim=-1)
    dataset_train = Data.TensorDataset(feature_con, label_con)
    train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=128, num_workers=0,
                                       shuffle=True)

    dataset_T, dataset_F = eval(netGG, dl_train_per_f)
    acc_f_gm = 100.0 * len(dataset_T) / (len(dataset_T) + len(dataset_F))
    td = get_dataloader(opt, False)
    print("       ")
    acc_test_gm = eval(netGG, td, False)
    print(acc_test_gm)
    eval(netGG, dl_train_per_f, False)

    preds_test_R = get_feature(netG, dataset_per_nf, logits=False)
    preds_test_R = torch.nn.functional.normalize(preds_test_R, p=2, dim=1)
    from util import getF
    label_test = getF(netGG, dl_train_per_f)
    print(torch.sum(label_test))
    eval(netGG, dl_train_per_f)



    ax = plt.axes(projection='3d')

    ax.scatter3D(d1_con[:lenth], d2_con[:lenth], d3_con[:lenth], cmap='g')

    ax.scatter3D(d1_con[lenth:], d2_con[lenth:], d3_con[lenth:], cmap='r')

    #plt.show()

    preds_test = get_feature(netG, dataset_per_f, logits=False)

    loss_test = eval_rl(netO, dl_train_per_f)

    d_test = []
    for i in range(len(preds_test)):
        preds_t = preds_test[i].unsqueeze(0)
        preds_t = preds_t.repeat(len(preds_test_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_test_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_test.append(dis)
    d_test = torch.stack(d_test)

    adro_test = eval_adverarial_attack(netG, dl_train_per_f, eps=8 / 255)


    d_test_1 = d_test.cpu()
    d_test_2 = adro_test.cpu()
    d_test_3 = loss_test.cpu()

    ind = torch.argsort(d_test_1, descending=True)
    max_1 = d_test_1[ind[3]]
    for i in range(3):
        d_test_1[ind[i]] = max_1

    max, min = torch.max(d_test_1), torch.min(d_test_1)
    d_test_1 = (d_test_1 - min) / (max - min)

    max, min = torch.max(d_test_2), torch.min(d_test_2)
    d_test_2 = (d_test_2 - min) / (max - min)

    max, min = torch.max(d_test_3), torch.min(d_test_3)
    d_test_3 = (d_test_3 - min) / (max - min)

    d1_test_con = d_test_1
    d2_test_con = d_test_2
    d3_test_con = d_test_3

    feature_test_con = torch.stack((d1_test_con.cpu(), d2_test_con.cpu(), d3_test_con.cpu()), dim=-1)
    label_test_con = label_test

    dataset_test = Data.TensorDataset(feature_test_con, label_test_con)
    test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=0,
                                           shuffle=False)


    for epoch in range(200):
        print("epoch is {}".format(epoch))
        trainb(model , optimizer,  train_dl, opt)
        if epoch % 1 == 0:
            acc_clean, acc_t, acc_f = evalb(model, train_dl, False)
            acc_clean2, acc_t2, acc_f2 = evalb(model, test_dl, False)

            path = './model_class' + str(opt.target_label) + '_cifar10.pth'

            torch.save(model.state_dict(), path)
    import csv
    with open('model.csv', 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(
            [opt.target_label, acc_clean, float(acc_t), float(acc_f), acc_clean2, float(acc_t2),
             float(acc_f2)])


    Flag = evalb(model, test_dl)

    Flag = np.array(Flag)

    dataset_per_f1 = Custom_dataset(dataset_per_f)
    ds_gT, ds_gF = dataset_per_f1.spilt(Flag)

    ds_gT = Custom_dataset(ds_gT)
    ds_gF = Custom_dataset(ds_gF)
    dataset_per_nf = Custom_dataset(dataset_per_nf)

    length = len(dataset_per_nf)
    torch.manual_seed(0)
    train_size, validate_size = int(0.3 * length), length - int(0.3 * length)
    dataset_per_nf, _ = torch.utils.data.random_split(dataset_per_nf, [train_size, validate_size])
    dataset_per_nf = Custom_dataset(dataset_per_nf)
    print(len(dataset_per_nf))

    length = len(dataset_t_r)
    torch.manual_seed(0)
    train_size, validate_size = int(0.3 * length), length - int(0.3 * length)
    dataset_t_r, _ = torch.utils.data.random_split(dataset_t_r, [train_size, validate_size])
    dataset_t_r = Custom_dataset(dataset_t_r)

    #dataset_gcon = ConcatDataset([ds_gF, ds_gT, dataset_per_nf, dataset_t_r])
    dataset_gcon = ConcatDataset([ds_gF,  dataset_per_nf, dataset_t_r])

    dl_train_gcon = torch.utils.data.DataLoader(dataset_gcon, batch_size=128, num_workers=opt.num_workers, shuffle=True)
    dl_train_r = torch.utils.data.DataLoader(dataset_t_r, batch_size=128, num_workers=opt.num_workers,
                                              shuffle=False)
    td = get_dataloader(opt, False)

    eval(netG, td)

    best_acc_r = 0
    best_acc_f = 0
    ar = 0
    gold_acc_f = (len(ds_gT) / (len(ds_gT) + len(ds_gF))) * 100.0

    optimizerC = torch.optim.Adam(netG.parameters(), lr=0.00005)


    for epoch in range(10):
        print("Epoch {}:".format(epoch + 1))

        # ds_gF.changelabel(100)
        ds_gF.changelabel(1)
        ds_gT.changelabel(0)
        dataset_per_nf.changelabel(0)
        dataset_t_r.changelabel(0)

        # class_un(netC, optimizerC, schedulerC, dl_train_gcon, opt.target_label)
        # ds_gF.changelabel(opt.target_label)
        kl_un(netG, netGT, netUL, optimizerC, dl_train_gcon)

        ds_gF.resetlabel()
        ds_gT.resetlabel()
        dataset_per_nf.resetlabel()
        dataset_t_r.resetlabel()

        if epoch % 1 == 0:

            accF = eval(netG, dl_train_per_f, False)
            accR = eval(netG, td, False)
            accr = eval(netG, dl_train_r, False)

            best_acc_f = accF
            best_acc_r = accR
            ar = accr


    path = './model_class' + str(opt.target_label) + '_cifar10_unlearned.pth'

    torch.save(netG.state_dict(), path)

    import csv
    with open('data.csv', 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(
            [opt.target_label, float(acc_f_gm), float(acc_test_gm), float(best_acc_f), float(best_acc_r), float(ar),
             len(ds_gT), len(ds_gF)])


if __name__ == "__main__":
    main()