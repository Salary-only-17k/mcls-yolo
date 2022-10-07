import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as DataLoader
import time
import copy
import datetime as dt

from nets.factory_nets import all_net
from utils.getdata import Dataset_1head
from utils.common import get_size, show_db, show_lg, show_er, check_save_pth
from utils.parse_cfg import parse_opt


__doc__ = \
    """
一个分类头
"""

'''
if torch.cuda.is_available():
    device = torch.device(f"cuda:{device_n}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
'''


opts = parse_opt()
opts.adam = True
opts.train_mode = 0 # {1 normal-train }
"""
"yolov5_62cls_st":[yolov5n_cls_st,yolov5s_cls_st,yolov5l_cls_st,yolov5m_cls_st],
"yolov5_62cls_dw":[yolov5n_cls_dw,yolov5s_cls_dw,yolov5l_cls_dw,yolov5m_cls_dw],
"yolov5_62cls_dp":[yolov5n_cls_dp,yolov5s_cls_dp,yolov5l_cls_dp,yolov5m_cls_dp],
"""
Net = all_net['yolov5_62cls_st'][0]


def run(opts):
    mode_Lst = opts.mode_Lst[:2]
    dataDataset = {mode: Dataset_1head(mode, opts.data_path, opts.data2index[0]) for mode in mode_Lst}
    dataloader = {mode: DataLoader(dataDataset[mode], batch_size=opts.batch_size * len(opts.n_cuda), shuffle=True, \
                                   num_workers=opts.workers, drop_last=True) for mode in mode_Lst}

    show_lg(f'Dataset loaded! length of train set is ', len(dataloader["train"]))
    data_size = {v: len(dataDataset[v]) for v in opts.mode_Lst}
    show_lg('data_size',data_size)

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(num_cls1=opts.n_cls[0]).run()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=opts.n_cuda)
    model.to(device=opts.device)
    show_lg('model using device ', opts.device)
    show_lg('parallel mulit device ', opts.n_cuda)

    if opts.resume_weights:
        model.load_state_dict(torch.load(opts.resume_weights)['model'].state_dict())

    criterion = nn.CrossEntropyLoss()
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opts.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    show_lg('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)),'')
    del pg0, pg1, pg2
    stz = max(round(192/opts.batch_size),1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stz, gamma=0.1)
    show_lg("train cfg:", opts)
    log_pth = check_save_pth(opts.log_name)
    with SummaryWriter(log_dir=log_pth) as writer:
        train_loop(model, dataloader, criterion, optimizer, exp_lr_scheduler, writer, data_size, opts)
        test_loop(model, dataloader, data_size, writer)
def train_loop(model, dataloader, criterion, optimizer, exp_lr_scheduler, writer, data_size, opts):
    since = time.time()
    if opts.train_mode  == 0:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for nepoch in range(opts.epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:  # val
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for data, label in tqdm.tqdm(dataloader[phase], desc=f'{nepoch + 1}/{opts.epochs}'):
                    data = data.cuda(device=opts.n_cuda[-1])
                    label = label.cuda(device=opts.n_cuda[-1])
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        with amp.autocast():
                            out = model(data)
                            _, pre = torch.max(out, 1)
                            loss = criterion(out, label.squeeze())
                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
                epoch_loss = running_loss / data_size[phase]
                epoch_acc = running_corrects.double() / data_size[phase]
                writer.add_scalar(f"{phase}/Loss", epoch_loss, nepoch)
                writer.add_scalar(f"{phase}/Acc", epoch_acc, nepoch)
                show_lg(f'\n{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}', '\n')
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            exp_lr_scheduler.step()
            time_elapsed = time.time() - since
            show_lg('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60), '')
        writer.add_graph(model, data)
        show_lg('Best val Acc: {:4f}'.format(best_acc), '')
        model.load_state_dict(best_model_wts)
        torch.save({"model": model, "acc": best_acc},
                   f"{opts.log_pth}/{model.get_name()}_Valacc_{best_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt")
    elif opts.train_mode  == 1:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for nepoch in range(opts.epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:  # val
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                for data, label in tqdm.tqdm(dataloader[phase], desc=f'{nepoch + 1}/{opts.epochs}'):
                    data = data.cuda(device=opts.n_cuda[-1])
                    label = label.cuda(device=opts.n_cuda[-1])
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        with amp.autocast():
                            out = model(data)
                            _, pre = torch.max(out, 1)
                            loss = criterion(out, label.squeeze())
                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
                epoch_loss = running_loss / data_size[phase]
                epoch_acc = running_corrects.double() / data_size[phase]
                writer.add_scalar(f"{phase}/Loss", epoch_loss, nepoch)
                writer.add_scalar(f"{phase}/Acc", epoch_acc, nepoch)
                show_lg(f'\n{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}', '\n')
                if phase == 'val' and epoch_acc >= opts.early_lst["acc"]:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            exp_lr_scheduler.step()
            time_elapsed = time.time() - since
            show_lg('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60), '')
        writer.add_graph(model, data)
        show_lg('Best val Acc: {:4f}'.format(best_acc), '')
        model.load_state_dict(best_model_wts)
        torch.save({"model": model, "acc": best_acc},
                   f"{opts.log_pth}/{model.get_name()}_Valacc_{best_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt")
    elif opts.train_mode == 2:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for nepoch in range(opts.epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:  # val
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for data, label in tqdm.tqdm(dataloader[phase], desc=f'{nepoch + 1}/{opts.epochs}'):
                    data = data.cuda(device=opts.n_cuda[-1])
                    label = label.cuda(device=opts.n_cuda[-1])
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        with amp.autocast():
                            out = model(data)
                            _, pre = torch.max(out, 1)
                            loss = criterion(out, label.squeeze())
                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
                epoch_loss = running_loss / data_size[phase]
                epoch_acc = running_corrects.double() / data_size[phase]
                writer.add_scalar(f"{phase}/Loss", epoch_loss, nepoch)
                writer.add_scalar(f"{phase}/Acc", epoch_acc, nepoch)
                show_lg(f'\n{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}', '\n')
                if phase == 'val' and epoch_loss <= opts.early_lst["loss"]:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            exp_lr_scheduler.step()
            time_elapsed = time.time() - since
            show_lg('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60), '')
        writer.add_graph(model, data)
        show_lg('Best val Acc: {:4f}'.format(best_acc), '')

        lastpth = f"{opts.log_pth}/{model.get_name()}_lastacc_{epoch_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt"
        torch.save({"model": model, "acc": best_acc},lastpth)
        bestpth  = f"{opts.log_pth}/{model.get_name()}_bestacc_{best_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt"
        model.load_state_dict(best_model_wts)
        torch.save({"model": model, "acc": best_acc},bestpth )
        show_lg(f"best-model save to : {bestpth}")
        show_lg(f"last-model save to : {lastpth}")
    else:
        show_db("flag value is error", '')


def test_loop(model, dataloader, data_size, writer):
    model.eval()
    running_corrects = 0
    for data, label in dataloader['test']:
        data = data.to(opts.device)
        label = label.to(opts.device)
        with torch.set_grad_enabled(False):
            out = model(data)
            _, pre = torch.max(out, 1)
        running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
    epoch_acc = running_corrects.double() / data_size['test']
    show_lg(f'test  Acc-Non: {epoch_acc:.4f}')
if __name__ == "__main__":
    run(opts)
