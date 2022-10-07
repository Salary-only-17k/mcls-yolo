import tqdm
import time
import copy
import datetime as dt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.getdata import (DataLoaderX,Dataset_3head,Dataset_2head,Dataset_1head)
from nets.factory_nets import all_net 
from utils.losses import MulitLoss
from utils.parse_cfg import parse_opt
from utils.common import show_lg,show_db

opts = parse_opt()
opts.pth_A = r''
opts.pth_B = r''
opts.pth_C = r''
opts.mode_Lst = ['train', 'val', 'test']
opts.workers = 0  # PyTorch读取数据线程数量
opts.epochs = 30
opts.batch_size=32
opts.resume_weights = r''
opts.save_model_path = r''
best_acc = opts.early_lst['acc']


num_c_1 = 11
num_c_2 = 2
num_c_3 = 3
lr = 1e-5
epoches = opts.epochs
Net = all_net['res50_3h_n'](num_classes1=num_c_1, num_classes2=num_c_2,num_classes3=num_c_3)

'''
if torch.cuda.is_available():
    device = torch.device(f"cuda:{device_n}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
'''

def run_earlyacc():
    show_lg("train cfg:", opts)
    dataDataset = {mode: Dataset_3head(mode, opts.data_path) for mode in opts.mode_Lst} #!
    dataloader = {mode: DataLoaderX(dataDataset[mode](batch_size=opts.batch_size, shuffle=True,  \
                                    num_workers=opts.workers,drop_last=True)) \
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
    m = Net.to(device=opts.device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()
    amp = True if opts.device!='cpu' else False
    scaler1 = torch.cuda.amp.GradScaler(enabled=amp )
    scaler2 = torch.cuda.amp.GradScaler(enabled=amp)
    scaler3 = torch.cuda.amp.GradScaler(enabled=amp)
    scalers_lst = [scaler1,scaler2,scaler3]
    criterions_lst = [criterion1, criterion2, criterion3]
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    with SummaryWriter(log_dir=opts.log_path) as writer:
        train_loop(dataloader, m, criterions_lst, scalers_lst,optimizer, exp_lr_scheduler, data_size, writer, amp)
        test_loop(dataloader, m,data_size, writer)


def train_loop(dataloader, model, criterions_lst,scalers_lst,optimizer, exp_lr_scheduler, data_size, writer,amp):
    '''
    :param flag: 1 早停， 2 保存最优解
    :return:
    '''
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for nepoch in range(epoches):
        show_lg('Epoch', f'{nepoch}/{epoches - 1}')
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
            for data, [labela, labelb, labelc] in tqdm.tqdm(dataloader[phase],desc=f'Epoch {nepoch+1}/{epoches}'):
                num += 1
                # print([labela, labelb])
                data = data.to(opts.device)
                labela = labela.to(opts.device)
                labelb = labelb.to(opts.device)
                labelc = labelc.to(opts.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=amp):
                    outa, outb, outc = model(data)
                    # print(f"aouta  outb size : {outa.size(),outb.size()}")
                    _, pre_a = torch.max(outa, 1)
                    _, pre_b = torch.max(outb, 1)
                    _, pre_c = torch.max(outc, 1)
                    loss1 = criterions_lst[0](outa, labela.squeeze())
                    loss2 = criterions_lst[1](outb, labelb.squeeze())
                    loss3 = criterions_lst[1](outc, labelc.squeeze())
                    if phase == 'train':
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)
                        loss3.backward(retain_graph=True)
                        optimizer.step()

                running_loss1 += loss1.item() * data.size(0)
                running_loss2 += loss2.item() * data.size(0)
                running_loss3 += loss3.item() * data.size(0)
                running_corrects_1 += torch.sum(pre_a.view(opts.batch_size, -1) == labela.data)
                running_corrects_2 += torch.sum(pre_b.view(opts.batch_size, -1) == labelb.data)
                running_corrects_3 += torch.sum(pre_c.view(opts.batch_size, -1) == labelc.data)
                
            # show_db('num loop', num)
            epoch_loss1 = running_loss1 / data_size[phase]
            epoch_loss2 = running_loss2 / data_size[phase]
            epoch_loss3 = running_loss2 / data_size[phase]
            epoch_acc_1 = running_corrects_1.double() / data_size[phase]
            epoch_acc_2 = running_corrects_2.double() / data_size[phase]
            epoch_acc_3 = running_corrects_3.double() / data_size[phase]
            writer.add_scalar(f"{phase}/Loss_1head", epoch_loss1)
            writer.add_scalar(f"{phase}/Acc_1head", epoch_acc_1)
            writer.add_scalar(f"{phase}/Loss_2head", epoch_loss2)
            writer.add_scalar(f"{phase}/Acc_2head", epoch_acc_2)
            writer.add_scalar(f"{phase}/Loss_3head", epoch_loss3)
            writer.add_scalar(f"{phase}/Acc_3head", epoch_acc_3)
            print(
                f'Epoch[{nepoch+1}/{epoches}] {phase} \
                    Loss-1: {epoch_loss1:.4f}  Loss-2: {epoch_loss2:.4f}  Loss-3: {epoch_loss3:.4f} \
                    Acc-1: {epoch_acc_1:.4f}  Acc-2: {epoch_acc_2:.4f}  Acc-3: {epoch_acc_3:.4f} ')
            tmp_acc = (1/3 * (epoch_acc_1 + epoch_acc_2+epoch_acc_3)) 
            if phase == 'val' and tmp_acc > best_acc:
                best_acc = tmp_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        exp_lr_scheduler.step()
        time_elapsed = time.time() - since
        show_lg('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    show_lg('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save({"model": model, "acc": best_acc},
                f"{opts.save_model_path}/{model.get_name()}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt")



def test_loop(dataloader, model, data_size, writer):
    model.eval()
    running_corrects_1 = 0
    running_corrects_2 = 0
    running_corrects_3 = 0
    for data, [labela, labelb, labelc] in dataloader['test']:
        data = data.to(opts.device)
        labela = labela.to(opts.device)
        labelb = labelb.to(opts.device)
        labelc = labelc.to(opts.device)
        with torch.set_grad_enabled(False):
            outa, outb, outc = model(data)
            _, pre_a = torch.max(outa, 1)
            _, pre_b = torch.max(outb, 1)
            _, pre_c = torch.max(outc, 1)
        running_corrects_1 += torch.sum(pre_a.view(opts.batch_size, -1) == labela.data)
        running_corrects_2 += torch.sum(pre_b.view(opts.batch_size, -1) == labelb.data)
        running_corrects_3 += torch.sum(pre_c.view(opts.batch_size, -1) == labelc.data)
    epoch_acc_1 = running_corrects_1.double() / data_size['test']
    epoch_acc_2 = running_corrects_2.double() / data_size['test']
    epoch_acc_3 = running_corrects_3.double() / data_size['test']
    writer.add_graph(model, data)
    show_lg(f'test  Acc-1: {epoch_acc_1:.4f}  Acc-2: {epoch_acc_2:.4f}  Acc-3: {epoch_acc_3:.4f}')
    return model
