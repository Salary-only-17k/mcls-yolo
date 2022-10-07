import torch.nn as nn
import pprint
import torch
import sys
sys.path.append("..")
from nets.utils.layer_tools_dp import C3,Conv,Classify
from nets.utils.struct_cfg import yolov5_cfg

class yolov5_cls_sm_dp(nn.Module):
    def __init__(self,
                 cfg:dict,
                 flg:str='s'
                 ):
        """
            conv使用的正常卷积
        """
        super(yolov5_cls_sm_dp, self).__init__()
        self.cfg = cfg
        self.flg = flg
        if self.flg =='s':
            self.feature_ = self._build_feature_map_1()
            self.cls_1 = self._build_cls_head_1('1')
            self.cls_2 = self._build_cls_head_1('2')
            self.cls_3 = self._build_cls_head_1('3')
        elif self.flg=='m':
            self.feature_ = self._build_feature_map_2()
            self.cls_1 = self._build_cls_head_2('1')
            self.cls_2 = self._build_cls_head_2('2')
            self.cls_3 = self._build_cls_head_2('3')
        else:
            raise RuntimeError
    def _build_feature_map_1(self):
        feature_map = nn.ModuleList()
        # feature_map = []
        feature_map_cfg = self.cfg['feature_map']
        
        feature_map.append(Conv(*feature_map_cfg[0][1]))    
        feature_map.append(Conv(*feature_map_cfg[1][1])) 
        feature_map.extend([C3(*feature_map_cfg[2][1])]*feature_map_cfg[2][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[3][1])) 
        feature_map.extend([C3(*feature_map_cfg[4][1])]*feature_map_cfg[4][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[5][1])) 
        feature_map.extend([C3(*feature_map_cfg[6][1])]*feature_map_cfg[6][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[7][1])) 
        feature_map.extend([C3(*feature_map_cfg[8][1])]*feature_map_cfg[8][0][0]) 
      
        return nn.Sequential(*feature_map)
    def _build_cls_head_1(self,indx):
        classify = nn.Sequential()
        classify_cfg = self.cfg['classify']
        classify.add_module(f'cls_{indx}',Classify(*classify_cfg[indx]))
        return classify
    
    def _build_feature_map_2(self):
        feature_map = nn.ModuleList()
        feature_map_cfg = self.cfg['feature_map']
        feature_map.append(Conv(*feature_map_cfg[0][1]))    
        feature_map.append(Conv(*feature_map_cfg[1][1])) 
        feature_map.extend([C3(*feature_map_cfg[2][1])]*feature_map_cfg[2][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[3][1])) 
        feature_map.extend([C3(*feature_map_cfg[4][1])]*feature_map_cfg[4][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[5][1])) 
        feature_map.extend([C3(*feature_map_cfg[6][1])]*feature_map_cfg[6][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[7][1])) 
        last_n = feature_map_cfg[8][0][0]
        if last_n>1:
            feature_map.extend([C3(*feature_map_cfg[8][1])]*(last_n-1)) 
        return nn.Sequential(*feature_map)
    
    def _build_cls_head_2(self,indx):
        classify = nn.Sequential()
        feature_map_cfg = self.cfg['feature_map']
        classify_cfg = self.cfg['classify']
        classify.add_module(f'cls_C3_{indx}',C3(*feature_map_cfg[8][1])) 
        classify.add_module(f'cls_fc_{indx}',Classify(*classify_cfg[indx]))
        return classify

    def forward(self,x):
        x = self.feature_(x)
        out1 = self.cls_1(x)
        out2 = self.cls_2(x)
        out3 = self.cls_3(x)
        return [out1,out2,out3]

class yolov5_cls_l_dp(nn.Module):
    def __init__(self,
                 cfg:dict,
                 ):
        super(yolov5_cls_l_dp, self).__init__()
        self.cfg = cfg
        self.decode1 = self._build_decode()
        self.decode2 = self._build_decode()
        self.decode3 = self._build_decode()
        self.feature_ = self._build_feature_map_1()
        self.cls_1 = self._build_cls_head_1('1')
        self.cls_2 = self._build_cls_head_1('2')
        self.cls_3 = self._build_cls_head_1('3')
    def _build_decode(self):
        decode = nn.ModuleList()
        feature_map_cfg = self.cfg['feature_map']
        decode.append(Conv(*feature_map_cfg[0][1]))    
        decode.append(Conv(*feature_map_cfg[1][1])) 
        return nn.Sequential(*decode)

    def _build_feature_map_1(self):
        feature_map = nn.ModuleList()
        feature_map_cfg = self.cfg['feature_map']
        feature_map.extend([C3(*feature_map_cfg[2][1])]*feature_map_cfg[2][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[3][1])) 
        feature_map.extend([C3(*feature_map_cfg[4][1])]*feature_map_cfg[4][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[5][1])) 
        feature_map.extend([C3(*feature_map_cfg[6][1])]*feature_map_cfg[6][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[7][1])) 
        last_n = feature_map_cfg[8][0][0]
        if last_n>1:
            feature_map.extend([C3(*feature_map_cfg[8][1])]*(last_n-1))
        return nn.Sequential(*feature_map)
    def _build_cls_head_1(self,indx):
        classify = nn.Sequential()
        feature_map_cfg = self.cfg['feature_map']
        classify_cfg = self.cfg['classify']
        classify.add_module(f'cls_C3_{indx}',C3(*feature_map_cfg[8][1])) 
        classify.add_module(f'cls_{indx}',Classify(*classify_cfg[indx]))
        return classify

    def forward(self,x):
        x1 = self.decode1(x[0])
        x2 = self.decode2(x[1])
        x3 = self.decode3(x[2])
        x = torch.mul(x3,torch.mul(x1,x2))
        x = self.feature_(x)
        out1 = self.cls_1(x)
        out2 = self.cls_2(x)
        out3 = self.cls_3(x)
        return [out1,out2,out3]

#~ s meaning small-struct
def yolov5n_cls_s_dp(num_cls1,num_cls2,num_cls3): 
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_n()
    return yolov5_cls_sm_dp(cfg,'s')
def yolov5s_cls_s_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_s() 
    return yolov5_cls_sm_dp(cfg,'s')

def yolov5l_cls_s_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_l()
    return yolov5_cls_sm_dp(cfg,'s')
def yolov5m_cls_s_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_m()
    return yolov5_cls_sm_dp(cfg,'s')


#~ m meaning middle-struct
def yolov5n_cls_m_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_n()
    return yolov5_cls_sm_dp(cfg,'m')

def yolov5s_cls_m_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_s()
    return yolov5_cls_sm_dp(cfg,'m')

def yolov5l_cls_m_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_l()
    return yolov5_cls_sm_dp(cfg,'m')

def yolov5m_cls_m_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_m()
    return yolov5_cls_sm_dp(cfg,'m')


#~ l meaning large-struct
def yolov5n_cls_l_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_n()
    return yolov5_cls_l_dp(cfg)

def yolov5s_cls_l_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_s()
    return yolov5_cls_l_dp(cfg)


def yolov5l_cls_l_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_l()
    return yolov5_cls_l_dp(cfg)
def yolov5m_cls_l_dp(num_cls1,num_cls2,num_cls3):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=num_cls2,num_classes3=num_cls3).yolov5_m()
    return yolov5_cls_l_dp(cfg)



if __name__ == "__main__":
    import os
    func = yolov5s_cls_s_dp
    pth = "tmp.pt"

    data = torch.randn(1,3,224,224)
    model = func(4,3,2)
    pprint.pprint(model(data))
    torch.save(model,pth)    
    print(f"{pth}    {os.stat(pth).st_size/1024/1024:.4f} M")
