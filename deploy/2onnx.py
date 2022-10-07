from dis import show_code
from multiprocessing.sharedctypes import Value
from turtle import forward
import torch
import torch.nn as nn
import onnx
import argparse
import sys,os
sys.path.append("../..")


from utils.common import show_lg,show_db
from nets.yolov5_62cls_st import yolov5n_cls_s,yolov5s_cls_s,yolov5l_cls_s,yolov5m_cls_s, \
                                  yolov5n_cls_m,yolov5s_cls_m,yolov5l_cls_m,yolov5m_cls_m, \
                                  yolov5n_cls_l,yolov5s_cls_l,yolov5l_cls_l,yolov5m_cls_l
from nets.yolov5_62cls_dp import yolov5n_cls_s_dp,yolov5s_cls_s_dp,yolov5l_cls_s_dp,yolov5m_cls_s_dp, \
                                  yolov5n_cls_m_dp,yolov5s_cls_m_dp,yolov5l_cls_m_dp,yolov5m_cls_m_dp, \
                                  yolov5n_cls_l_dp,yolov5s_cls_l_dp,yolov5l_cls_l_dp,yolov5m_cls_l_dp
from nets.yolov5_62cls_dw import yolov5n_cls_s_dw,yolov5s_cls_s_dw,yolov5l_cls_s_dw,yolov5m_cls_s_dw, \
                                  yolov5n_cls_m_dw,yolov5s_cls_m_dw,yolov5l_cls_m_dw,yolov5m_cls_m_dw, \
                                  yolov5n_cls_l_dw,yolov5s_cls_l_dw,yolov5l_cls_l_dw,yolov5m_cls_l_dw
from nets.yolov5_62cls import yolov5n_cls_st,yolov5s_cls_st,yolov5l_cls_st,yolov5m_cls_st, \
                                  yolov5n_cls_dw,yolov5s_cls_dw,yolov5l_cls_dw,yolov5m_cls_dw, \
                                  yolov5n_cls_dp,yolov5s_cls_dp,yolov5l_cls_dp,yolov5m_cls_dp

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_pth', type=str, default='', help='pt weight path')
    parser.add_argument('--onnx_pth', type=str,default='',help='onnx file path')
    parser.add_argument('--net_name', type=str,default='',help='net name')
    parser.add_argument('--cls_num', type=int,nargs='+', default='',help='classify head num')
    parser.add_argument('--dynamic', type=float, default=0.1, help='split train-data val-data test-data ratio')
    opts = parser.parse_args()
    if isinstance(opts.cls_num,int):
        opts.cls_num = [opts.cls_num]
    if opts.onnx_pth is "":
        opts.onnx_pth = opts.weights_pth.replace(".pt",".oonx")
    show_lg('opts: ',opts)
    return opts


def load_weights(model_name,pth,cls_lst):
    base_model = model(*cls_lst)
    model = eval(model_name)
    if os.path.exists(pth):
        base_model.load_state_dict(torch.load(pth,map_location='cpu')['model'].state_dict())
    else:
        show_db("DEBUG MODE",'')
    return base_model

class per_1model3_pos(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.backbone = model
    def forward(self,x):
        x=torch.divide(x,255.)
        outs_lst = self.backbone(x)
        out_1 = torch.max(outs_lst[0],1).values
        out_2 = torch.max(outs_lst[1],1).values
        out_3 = torch.max(outs_lst[2],1).values
        return [out_1,out_2,out_3]

class per_1model1_pos(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.backbone = model
    def forward(self,x):
        x=torch.divide(x,255.)
        out = self.backbone(x)
        out = torch.max(out,1).values
        return out

class per_3model3_pos(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.backbone = model
    def forward(self,x):
        x=torch.divide(x,255.)
        outs_lst = self.backbone(x[0],x[1],x[2])
        out_1 = torch.max(outs_lst[0],1).values
        out_2 = torch.max(outs_lst[1],1).values
        out_3 = torch.max(outs_lst[2],1).values
        return [out_1,out_2,out_3]

def tonnx(model,onnx_pth):
    data = torch.randn(1,3,224,224).cpu()
    inptn = ['data1']
    outptn = ['out1','out2','out3']
    torch.onnx.export(model,data,'propost.onnx',input_names=inptn,output_names=outptn)
    om = onnx.load(onnx_pth)
    onnx.checker.check_model(om)
    show_lg("onnx save to: ",onnx_pth)


def tonnx3(model,onnx_pth):
    data = torch.randn(1,3,224,224).cpu()
    inptn = ['data1']
    outptn = ['out1']
    torch.onnx.export(model,data,'propost.onnx',input_names=inptn,output_names=outptn)
    om = onnx.load(onnx_pth)
    onnx.checker.check_model(om)
    show_lg("onnx save to: ",onnx_pth)

def tonnx33(model,onnx_pth):
    data = torch.randn(1,3,224,224).cpu()
    inptn = ['data1','data2','data3']
    outptn = ['out1','out2','out3']
    torch.onnx.export(model,data,'propost.onnx',input_names=inptn,output_names=outptn)
    om = onnx.load(onnx_pth)
    onnx.checker.check_model(om)
    show_lg("onnx save to: ",onnx_pth)

if __name__=="__main__":
    opts = parse_opt()
    model = load_weights(opts.weights_pth,opts.net_name,opts.cls_num).cpu()
    model.eval()
    if len(opts.cls_num)==1:
        model = per_1model3_pos(model)
        tonnx(model,opts.onnx_pth)
    elif len(opts.cls_num)==3:
        model = per_1model1_pos(model)
        tonnx3(model,opts.onnx_pth)
    else:
        raise 'cls_num 的长度不对呀',ValueError
    