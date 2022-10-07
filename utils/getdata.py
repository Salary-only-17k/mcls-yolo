import os
import pathlib
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DataLoader



class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def transfroms(input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.CenterCrop((input_size, input_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

class Dataset_3head(data.Dataset): 
    def __init__(self, mode,input_size, dir:str,label2indx:dict,format:str='.jpg'):  
        """
        同源数据，一张图像对应多个标签
        Args:
            mode (_type_): _description_
            dir (str): _description_
            label2indx (dict): _description_  {"A":{"car":0,"bus":1,"truck":2},
                                                "B":{"red":0,"yellow":1,"white":2},
                                                "C":{"big":0,"little":1}}
            format (str, optional): _description_. Defaults to '.jpg'.
        """
        self.mode = mode
        self.label2indx = label2indx
        self.format =format
        self.list_img = []  
        self.list_label_a = [] 
        self.list_label_b = []  
        self.list_label_c = [] 
        self.data_size = 0  
        self.transform = transfroms(input_size)  # 转换方式

        if self.mode in ['train', 'val', 'test']:  #
            dir = dir + f'/{self.mode}/'  # 训练集路径在"dir"/train/
            for filepth in list(pathlib.Path(dir).glob("**/*{self.format}")):  # 遍历dir文件夹
                filepth = str(filepth)
                self.list_img.append(filepth)  # 将图片路径和文件名添加至image list
                self.data_size += 1  # 数据集增1
                filename = os.path.basename(filepth)
                label_pool = filename.split('_')
                label_A = label_pool[1]
                label_B = label_pool[2]
                label_C = label_pool[3]
                self.list_label_a.append(self.label2indx['A'][label_A]) 
                self.list_label_b.append(self.label2indx['B'][label_B])  
                self.list_label_c.append(self.label2indx['C'][label_C]) 
           
        else:
            print('Undefined Dataset!')
    def __doc__(self):
        print("0000n_Alabel_Blabel_Clabel_.jpg")

    def __getitem__(self, item):  # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':  # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])  # 打开图片
            labela = self.list_label_a[item]  # 获取image对应的label
            labelb = self.list_label_b[item]  # 获取image对应的label
            labelc = self.list_label_c[item]  # 获取image对应的label
            return self.transform['train'](img), \
                   [torch.LongTensor([labela]), torch.LongTensor([labelb]), torch.LongTensor([labelc])]  # 将image和label转换成PyTorch形式并返回
        elif self.mode in ['val', 'test']:  # 测试集只需读取image
            img = Image.open(self.list_img[item])  # 打开图片
            labela = self.list_label_a[item]  # 获取image对应的label
            labelb = self.list_label_b[item]  # 获取image对应的label
            labelc = self.list_label_c[item]  # 获取image对应的label
            return self.transform['test'](img), \
                   [torch.LongTensor([labela]), torch.LongTensor([labelb]), torch.LongTensor([labelc])]  # 将image和label转换成PyTorch形式并返回
        else:
            print('None')

    def __len__(self):
        return self.data_size  # 返回数据集大小


class Dataset_2head(data.Dataset): 
    def __init__(self, mode, dir:str,label2indx:dict,indx:list=[1,2],format:str='.jpg'):  
        self.mode = mode
        self.label2indx = label2indx
        self.format =format
        self.list_img = []  
        self.list_label_a = [] 
        self.list_label_b = []
        self.data_size = 0  
        self.transform = transfroms()  # 转换方式

        if self.mode in ['train', 'val', 'test']:  #
            dir = dir + f'/{self.mode}/'  # 训练集路径在"dir"/train/
            for filepth in list(pathlib.Path(dir).glob("**/*{self.format}")):  # 遍历dir文件夹
                filepth = str(filepth)
                self.list_img.append(filepth)  # 将图片路径和文件名添加至image list
                self.data_size += 1  # 数据集增1
                filename = os.path.basename(filepth)
                label_pool = filename.split('_')
                label_A = label_pool[indx[0]]
                label_B = label_pool[indx[1]]
    
                self.list_label_a.append(self.label2indx['A'][label_A]) 
                self.list_label_b.append(self.label2indx['B'][label_B])  
        
           
        else:
            print('Undefined Dataset!')
    def __doc__(self):
        print("0000n_Alabel_Blabel_Clabel_.jpg")

    def __getitem__(self, item):  # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':  # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])  # 打开图片
            labela = self.list_label_a[item]  # 获取image对应的label
            labelb = self.list_label_b[item]  # 获取image对应的label
       
            return self.transform['train'](img), \
                   [torch.LongTensor([labela]), torch.LongTensor([labelb])]  # 将image和label转换成PyTorch形式并返回
        elif self.mode in ['val', 'test']:  # 测试集只需读取image
            img = Image.open(self.list_img[item])  # 打开图片
            labela = self.list_label_a[item]  # 获取image对应的label
            labelb = self.list_label_b[item]  # 获取image对应的label
       
            return self.transform['test'](img), \
                   [torch.LongTensor([labela]), torch.LongTensor([labelb])]  # 将image和label转换成PyTorch形式并返回
        else:
            print('None')

    def __len__(self):
        return self.data_size  # 返回数据集大小

class Dataset_1head(data.Dataset): 
    def __init__(self, mode, dir:str,indx:int=1,format:str='.jpg'):  
        self.mode = mode
        self.format =format
        self.list_img = []  
        self.list_label= [] 
        self.transform = transfroms()  
        raise self.mode in ['train', 'val', 'test'],ValueError
        dir = dir + f'/{self.mode}/'  
        for filepth in list(pathlib.Path(dir).glob("**/*{self.format}")):  
            filepth = str(filepth)
            self.list_img.append(filepth)  
            filename = os.path.basename(filepth)
            label_pool = filename.split('_')
            label = label_pool[indx]
            self.list_label_a.append(float(label)) 
        self.data_size = len(self.list_label_a)           
       
    def __doc__(self):
        print("0000n_Alabel_Blabel_Clabel_.jpg")

    def __getitem__(self, item):  
        if self.mode == 'train':  
            img = Image.open(self.list_img[item]) 
            label = self.list_label[item]  
            return self.transform['train'](img),torch.LongTensor([label])  
        elif self.mode in ['val', 'test']:  
            img = Image.open(self.list_img[item])  
            labela = self.list_label[item]  
            return self.transform['test'](img), torch.LongTensor([label]) 
        else:
            print('None')

    def __len__(self):
        return self.data_size  # 返回数据集大小


# if __name__ == '__main__':
#     print(transfroms()['train'])
#     dataset_dir = r'../data'  # 数据集路径
#     # a, [b, c] = next(iter(MulitDataset('train', dataset_dir)))
#     # print(a)
#     # print(b)
#     # print(c)
#     print('-' * 20)
#     from torch.utils.data import DataLoader as DataLoader

#     # test_data = MulitDataset('train', dataset_dir)
#     test_data = HoTDataset('train', dataset_dir)
#     dataset = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

#     tmp = []
#     for d, l in dataset:
#         print(l)
#         tmp.append(len(d))
#     print(sum(tmp), "  ", len(dataset))

#     # a, [b, c] = next(iter(Dataset('test', dataset_dir)))
#     # print(a, b, c)
#     # a,[b,c] =Dataset('train', dataset_dir)
#     # print(a,b,c)
