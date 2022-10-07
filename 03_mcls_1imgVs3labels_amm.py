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
模仿hybridnet的训练方式，分头头分别训练
多头分类模型
一个图像对应3个标签
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
    dataDataset = {mode: Dataset_1head(mode, opts.data_path,indx=opts.head_indx) for mode in opts.mode_Lst} #!
    dataloader = {mode: DataLoaderX(dataDataset[mode],batch_size=opts.batch_size, shuffle=True,  num_workers=opts.workers,drop_last=True) \
                     for mode in opts.mode_Lst}
    show_lg('Dataset loaded! length of train set is {0}'.format(len(dataloader['train'])), '')
    data_size = {v: len(dataDataset[v]) for v in opts.mode_Lst}
    # show_lg('data_size : ', data_size)
    if torch.cuda.is_available():
        opts.device = torch.device(f"cuda:{opts.n_cuda}")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True  # 别注释开了 报错
    else:
        opts.device = torch.device("cpu")

    show_lg('using device ', opts.device)
    Net = all_net[opts.net][opts.netIndx]
    model = Net.to(device=opts.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    if opts.head_indx==1:
        if opts.init_method ==1:    
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
    elif opts.init_method in [2,3]:
        if opts.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0032, betas=(0.843, 0.999))  # adjust beta1 to momentum
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0032, momentum=0.843 nesterov=True)
    else:
        raise KeyError
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    with SummaryWriter(log_dir=opts.log_path) as writer:
        train_loop(model,dataloader, criterion, optimizer, exp_lr_scheduler, data_size, writer, opts.head_indx)
        test_loop(model,dataloader,data_size, writer,opts.head_indx)

def train_loop(model,dataloader, criterion,optimizer, exp_lr_scheduler, data_size, writer,head_indx):
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
            running_loss = 0.0
            running_corrects = 0
            num = 1
            for iter_data in tqdm.tqdm(dataloader,desc=f'Epoch {nepoch+1}/{opts.epoches}'):
                
                data,label = iter_data
                data,label = data.to(opts.device),label.to(opts.device)
                num += 1
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(data)[head_indx-1]
                    # print(f"aouta  outb size : {outa.size(),outb.size()}")
                    _, pre_a = torch.max(out, 1)
                    loss = criterion(out, label.squeeze())
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(pre_a.view(opts.batch_size, -1) == label.data)

                
            # show_db('num loop', num)
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.double() / data_size[phase]
    
            writer.add_scalar(f"{phase}/Loss_{head_indx}head", epoch_loss,nepoch)
            writer.add_scalar(f"{phase}/Acc_{head_indx}head", epoch_acc,nepoch)
            
            print(
                f'\nEpoch[{nepoch+1}/{opts.epoches}] {phase} \
                    Loss-{head_indx}: {epoch_loss:.4f}  Acc-{head_indx}: {epoch_acc:.4f}  ')
        
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
    
    lastpth = f"{opts.log_pth}_{head_indx}head/last_ValAcc_{epoch_acc:.3f}.pt"
    torch.save({"model": model, "acc": epoch_acc},lastpth)
    bestpth  = f"{opts.log_pth}_{head_indx}head/best_ValAcc_{best_acc:.3f}.pt"
    model.load_state_dict(best_model_wts)
    torch.save({"model": model, "acc": best_acc},bestpth )
    
    show_lg(f"best-model save to : ",opts.log_pth)



def test_loop(model,dataloader, data_size,epoch_acc):
    model.eval()
    running_corrects = 0
    num = 0
    for [data, label] in dataloader['test']:
        data = data.to(opts.device)
        label = label.to(opts.device)
        with torch.set_grad_enabled(False):
            out = model(data)[epoch_acc-1]
            _, pre = torch.max(out, 1)
        running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data1)
        num+=1
    epoch_acc = running_corrects.double() / num
    show_lg(f'test  {epoch_acc}head-Acc: {epoch_acc:.4f} ')
    torch.cuda.empty_cache()


if __name__ =="__main__":
    assert sys.args[1] == 'amm','cfg is error'
    opts = parse_opt()
    run(opts)
