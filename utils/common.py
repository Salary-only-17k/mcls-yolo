
import os
import pathlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DataLoader

def check_last(pth):
    pth_lst = list(pathlib.Path(pth).glob("*.pt"))
    ctime = []
    for pth in pth_lst:
        pth = str(pth)
        ctime.append(os.stat(pth).st_ctime)
    return str(pth_lst[max(ctime)])

def check_save_pth(file_name):
    tmp = file_name.split('_')
    file_name = f"{tmp[0]}_1" if len(tmp)==1 else f"{tmp[0]}_{int(tmp[1])+1}"
    return os.path.join('runs/train',file_name)
def get_size(model,name):
    pth =f'{model}.pt'
    torch.save(model,pth)
    show_lg(name,f'{os.stat(pth).st_size/1024/1024:4,f}M')
    os.remove(pth)


def statistical_data(data:dict,save_pth):
    """
    data = [5, 6, 7, 8]
    labels = ["a", "b", "c", "d"]
    data   {"A":{"x":[],"y":[]},
            "B":{"x":[],"y":[]},
            "B":{"x":[],"y":[]},}
    """
    num_subimg = len(list(data.keys()))
    for i,nm, in enumerate(list(data.keys())):
        plt.subplot(2,2,i+1)
        plt.bar(range(len(data[nm]['y'])), data[nm]['y'])
        plt.xticks(range(len(data[nm]['y'])),data[nm]['x'])
        plt.title(nm)
    plt.savefig(f'{save_pth}/labels_num.jpg')


def show_db(n,v):
    print(f"\033[1;33m[DEBUG]    {n}      {v}\033[0m")

def show_lg(n,v):
    print(f"\033[1;34m[LOG]      {n}      {v}\033[0m")

def show_er(n,v):
    print(f"\033[1;31m[ERROR]    {n}      {v}\033[0m")
    raise RuntimeError
    
   
        
def freeze_some_layer(model,flag):
    model_name = model.get_name()
    if flag==1:
        if 'resnet' in model_name.lower():
            freeze=['fc2','f3']
            print("mdoel will freeze {freeze}")
            for k,v in model.name_parameters():
                v.requires_grad=True
                if any(x in k for x in freeze):
                    v.requires_grad=False
        elif 'densenet' in model_name.lower():
            freeze=['classifier2.','classifier3.']
            print("mdoel will freeze {freeze}")
            for k,v in model.name_parameters():
                v.requires_grad=True
                if any(x in k for x in freeze):
                    v.requires_grad=False
        else:
            raise ValueError
    elif flag==2:
        if 'resnet' in model_name.lower():
            freeze_not=['fc2']
            print("mdoel will not freeze {freeze_not}")
            for k,v in model.name_parameters():
                v.requires_grad=False
                if any(x in k for x in freeze_not):
                    v.requires_grad=True
        elif 'densenet' in model_name.lower():
            freeze_not=['classifier2.']
            print("mdoel will not freeze {freeze_not}")
            for k,v in model.name_parameters():
                v.requires_grad=False
                if any(x in k for x in freeze_not):
                    v.requires_grad=True
        else:
            raise ValueError
    elif flag==3:
        if 'resnet' in model_name.lower():
            freeze_not=['f3']
            print("mdoel will not freeze {freeze_not}")
            for k,v in model.name_parameters():
                v.requires_grad=False
                if any(x in k for x in freeze_not):
                    v.requires_grad=True
        elif 'densenet' in model_name.lower():
            freeze_not=['classifier3.']
            print("mdoel will not freeze {freeze_not}")
            for k,v in model.name_parameters():
                v.requires_grad=False
                if any(x in k for x in freeze_not):
                    v.requires_grad=True
        else:
            raise ValueError
    else:
        if 'resnet' in model_name.lower():
            initWeightsNormal(model)
        elif 'densenet' in model_name.lower():
            initWeightsKaiming(model)
        else:
            raise ValueError


def initWeightsKaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def initWeightsNormal(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)