import tqdm
import time
import copy
import datetime as dt
import sys

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as DataLoader

from utils.common import freeze_some_layer
from nets.factory_nets import all_net
from utils.getdata import (DataLoaderX,Dataset_3head,Dataset_2head,Dataset_1head)
from utils.losses import MulitLoss
from utils.common import get_size, show_db, show_lg, show_er, check_save_pth
from utils.parse_cfg import parse_opt
__doc__ = \
"""
多头分类模型
一个图像对应3个标签
三组数据同时加载进行训练，1个图像对应3个标签
"""


'''
if torch.cuda.is_available():
    device = torch.device(f"cuda:{device_n}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
'''

def run(opts):
    show_lg("train cfg:", opts)
    mode_Lst = opts.mode_Lst
    dataDataset = {mode: Dataset_3head(mode, opts.data_path_1) for mode in mode_Lst} #!
    dataloader = {mode: DataLoaderX(dataDataset[mode],batch_size=opts.batch_size, shuffle=True,  \
                                    num_workers=opts.workers,drop_last=True) \
                for mode in mode_Lst}
    len_data = len(dataloader['train'])
    max_iter = len_data
    show_lg('Dataset for 1-head loaded! length of train set is {0}'.format(len_data), '')
    data_1_size = {v: len(dataDataset[v]) for v in opts.mode_Lst}
    data_size =  {'1':data_1_size,'max':max_iter}
    show_lg('data_1_size : ', data_1_size)

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = all_net[opts.net][opts.netIndx]
    model = Net(*opts.n_cls)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=opts.n_cuda)
    model.to(device=opts.device)
    show_lg('model using device ', opts.device)
    show_lg('parallel mulit device ', opts.n_cuda)
    freeze_some_layer(model,opts.head_indx)

    if opts.resume_weights:
        model.load_state_dict(torch.load(opts.resume_weights)['model'].state_dict())
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    criterion_3 = nn.CrossEntropyLoss()
    criterion_lst = [criterion_1,criterion_2,criterion_3]
    if opts.init_method ==0:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if opts.adam:
            optimizer = torch.optim.Adam(pg0, lr=0.0032, betas=(0.843, 0.999))  # adjust beta1 to momentum
        else:
            optimizer = torch.optim.SGD(pg0, lr=0.0032, momentum=0.843 nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': 0.00036})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        show_lg('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)),'')
        del pg0, pg1, pg2
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    show_lg('using device ', opts.device)
    step_size = max(7*32/round(opts.batch_size),1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    with SummaryWriter(log_dir=opts.log_path) as writer:
        train_loop(model,dataloader, max_iter, \
            criterion_lst, optimizer, exp_lr_scheduler, data_size, writer)
        test_loop(model,dataloader, data_size)


def train_loop(model,dataloader, max_iter, \
                criterion_lst,optimizer, exp_lr_scheduler, data_size, writer):
    '''
    :param flag: 1 早停， 2 保存最优解
    :return:
    '''
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for nepoch in range(opts.epoches):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:  # val
                model.eval()
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_corrects_1 = 0
            running_corrects_2 = 0
            running_corrects_3 = 0
            num = 1
            for iter_data in tqdm.tqdm(dataloader,desc=f'Epoch {nepoch+1}/{opts.epoches}'):

                data,label_1,label_2,label_3 = iter_data
                data,label_1,label_2,label_3 = data.to(opts.device),label_1.to(opts.device),label_2.to(opts.device),label_3.to(opts.device)
                num += 1
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out_1, out_2, out_3 = model(data)
                    # print(f"aouta  outb size : {outa.size(),outb.size()}")
                    _, pre_a = torch.max(out_1, 1)
                    _, pre_b = torch.max(out_2, 1)
                    _, pre_c = torch.max(out_3, 1)
                    loss1 = criterion_lst[0](out_1, label_1.squeeze())
                    loss2 = criterion_lst[1](out_2, label_2.squeeze())
                    loss3 = criterion_lst[1](out_3, label_3.squeeze())
                    if phase == 'train':
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)
                        loss3.backward(retain_graph=True)
                        optimizer.step()

                running_loss1 += loss1.item() * data.size(0)
                running_loss2 += loss2.item() * data.size(0)
                running_loss3 += loss3.item() * data.size(0)
                running_corrects_1 += torch.sum(pre_a.view(opts.batch_size, -1) == label_1.data)
                running_corrects_2 += torch.sum(pre_b.view(opts.batch_size, -1) == label_2.data)
                running_corrects_3 += torch.sum(pre_c.view(opts.batch_size, -1) == label_3.data)

            # show_db('num loop', num)
            epoch_loss1 = running_loss1 / data_size[phase]
            epoch_loss2 = running_loss2 / data_size[phase]
            epoch_loss3 = running_loss2 / data_size[phase]
            epoch_acc_1 = running_corrects_1.double() / data_size[phase]
            epoch_acc_2 = running_corrects_2.double() / data_size[phase]
            epoch_acc_3 = running_corrects_3.double() / data_size[phase]
            writer.add_scalar(f"{phase}/Loss_1head", epoch_loss1,nepoch)
            writer.add_scalar(f"{phase}/Acc_1head", epoch_acc_1,nepoch)
            writer.add_scalar(f"{phase}/Loss_2head", epoch_loss2,nepoch)
            writer.add_scalar(f"{phase}/Acc_2head", epoch_acc_2,nepoch)
            writer.add_scalar(f"{phase}/Loss_3head", epoch_loss3,nepoch)
            writer.add_scalar(f"{phase}/Acc_3head", epoch_acc_3,nepoch)
            print(
                f'\nEpoch[{nepoch+1}/{opts.epoches}] {phase} \
                    Loss-1: {epoch_loss1:.4f}  Loss-2: {epoch_loss2:.4f}  Loss-3: {epoch_loss3:.4f} \
                    Acc-1: {epoch_acc_1:.4f}  Acc-2: {epoch_acc_2:.4f}  Acc-3: {epoch_acc_3:.4f} ')
            epoch_acc = (1/3 * (epoch_acc_1 + epoch_acc_2+epoch_acc_3))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        exp_lr_scheduler.step()
        time_elapsed = time.time() - since
        show_lg('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    show_lg('Best val Acc: {:4f}'.format(best_acc))
    writer.add_graph(model, data)

    show_lg('Best val Acc: {:4f}'.format(best_acc), '')

    lastpth = f"{opts.log_pth}/last_ValAcc_{epoch_acc:.3f}.pt"
    torch.save({"model": model, "acc": epoch_acc},lastpth)
    bestpth  = f"{opts.log_pth}/best_ValAcc_{best_acc:.3f}.pt"
    model.load_state_dict(best_model_wts)
    torch.save({"model": model, "acc": best_acc},bestpth )

    show_lg(f"best-model save to : ",opts.log_pth)



def test_loop(model,dataloader, data_size):
    model.eval()
    running_corrects_1 = 0
    running_corrects_2 = 0
    running_corrects_3 = 0
    num = 0
    for [data, label1,label2, label3] in dataloader['test']:
        data = data.to(opts.device)
        label1 = label1.to(opts.device)
        label2 = label2.to(opts.device)
        label3 = label3.to(opts.device)
        with torch.set_grad_enabled(False):
            out_1, out_2, out_3 = model(data)
            _, pre_1 = torch.max(out_1, 1)
            _, pre_2 = torch.max(out_2, 1)
            _, pre_3 = torch.max(out_3, 1)
        running_corrects_1 += torch.sum(pre_1.view(opts.batch_size, -1) == label1.data1)
        running_corrects_2 += torch.sum(pre_2.view(opts.batch_size, -1) == label2.data2)
        running_corrects_3 += torch.sum(pre_3.view(opts.batch_size, -1) == label3.data3)
        num+=1
    epoch_acc_1 = running_corrects_1.double() / num
    epoch_acc_2 = running_corrects_2.double() / num
    epoch_acc_3 = running_corrects_3.double() / num
    show_lg(f'test  head1-Acc: {epoch_acc_1:.4f}  head2-Acc: {epoch_acc_2:.4f}  head3-Acc: {epoch_acc_3:.4f}')
    torch.cuda.empty_cache()

if __name__ == "__main__":
    assert sys.args[1] == 'ssm','cfg is error'
    opts = parse_opt()
    run(opts)
