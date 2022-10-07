import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#!----------------------------------------

class Dynamic_balance_coefficient_1():
    def __init__(self,losses_lst):
        self.losses_lst = losses_lst
    def _sum(self):
        self.loss1 = np.sum(self.losses_lst[0].cpu().detach().numpy())
        self.loss2 = np.sum(self.losses_lst[1].cpu().detach().numpy())
        self.loss3 = np.sum(self.losses_lst[2].cpu().detach().numpy())
    def _mean(self):
        self.loss1 = np.mean(self.losses_lst[0].cpu().detach().numpy())
        self.loss2 = np.mean(self.losses_lst[1].cpu().detach().numpy())
        self.loss3 = np.mean(self.losses_lst[2].cpu().detach().numpy())    
    def _std(self):
        self.loss1 = np.std(self.losses_lst[0].cpu().detach().numpy())
        self.loss2 = np.std(self.losses_lst[1].cpu().detach().numpy())
        self.loss3 = np.std(self.losses_lst[2].cpu().detach().numpy())
    def cal_hard(self,flg):
        assert flg in ['sum','mean','std'], "flg must be in ['sum','mean','std']"
        if flg == "sum":
            self._sum()
        elif flg == "mean":
            self._mean()
        else:
            self._std()
        ratio1 = (self.loss2+self.loss3)/(self.loss1+self.loss2+self.loss3)
        ratio2 = (self.loss1+self.loss3)/(self.loss1+self.loss2+self.loss3)
        ratio3 = (self.loss1+self.loss2)/(self.loss1+self.loss2+self.loss3)
        if ratio1>=0.3 and ratio2>=0.3 and ratio3>=0.3:
            return self.losses_lst
        else:
            return [self.losses_lst[0]*ratio1,self.losses_lst[1]*ratio2,self.losses_lst[2]*ratio3]
    # def cal_soft(self,flg):
    #     assert flg in ['sum','mean','std'], "flg must be in ['sum','mean','std']"
    #     if flg == "sum":
    #         self._sum()
    #     elif flg == "mean":
    #         self._mean()
    #     else:
    #         self._std()
    #     ratio1 = (self.loss2+self.loss3)/(self.loss1+self.loss2+self.loss3)
    #     ratio2 = (self.loss1+self.loss3)/(self.loss1+self.loss2+self.loss3)
    #     ratio3 = (self.loss1+self.loss2)/(self.loss1+self.loss2+self.loss3)
    #     if ratio1>=0.3 and ratio2>=0.3 and ratio3>=0.3:
    #         return self.losses_lst
    #     else:
    #         v = np.array([self.losses_lst[0]*ratio1,self.losses_lst[1]*ratio2,self.losses_lst[2]*ratio3])
    #         indgt03 = np.nozero(v>0.3)


    
class MulitLoss(nn.Module):
    def __init__(self):
        super(MulitLoss, self).__init__()

    def forward(self, a, a_, b, b_):
        return self._forward_b(a, a_, b, b_)

    def _forward_a(self, a, a_, b, b_):
        _, ya = torch.max(a_)
        _, yb = torch.max(b_)
        loss1 = nn.CrossEntropyLoss()
        loss2 = nn.CrossEntropyLoss()
        return abs(a - ya) * loss1(b, b_) + abs(b - yb) * loss2(a, a_)

    def _forward_b(self, a, a_, b, b_):
        loss1 = nn.CrossEntropyLoss()
        loss2 = nn.CrossEntropyLoss()
        return 0.5*(loss1(b_,b) + loss2(a_, a))

    def _forward_c(self, a, a_, b, b_):
        loss1 = nn.CrossEntropyLoss()
        loss2 = nn.CrossEntropyLoss()
        return abs(a - a_) * loss1(a, a_) + abs(b - yb) * loss2(b, b_)



