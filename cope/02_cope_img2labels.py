import pathlib
import os
import tqdm
import pprint
from multiprocessing import Process 
import random
import shutil
import argparse
import datetime as dt
import sys
sys.path.append("../..")
from utils.common import show_lg,show_db
"""plain text
images\
        c\
            w\         
                h\     imgs...
                t\     imgs...
                u\      imgs...
            h\
                h\     imgs...
                t\     imgs...
                u\      imgs...
        b\
            w\       
                h\     imgs...
                t\     imgs...
                u\      imgs...
            h\
                h\     imgs...
                t\     imgs...
                u\      imgs...
        t\
            w\  
                h\     imgs...
                t\     imgs...
                u\      imgs...
            h\
                h\     imgs...
                t\     imgs...
                u\      imgs...
"""
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcDir', type=str, default='',required=True, help='base data')
    parser.add_argument('--resDir', type=str, default='', required=True,help='cope images, and copy to dir')
    parser.add_argument('--ratio', type=float, default=0.1, help='split train-data val-data test-data ratio')
    parser.add_argument('--format', nargs='+', type=str, default=['.jpg','.png'], help='image format')
    opts = parser.parse_args()
    opts.label2index = {"1":{"c":0,"b":1,"t":2},
                        '2':{'w':0,'h':1},
                        '3':{'h':0,'t':1,'u':2}}
    show_lg('opts: ',opts)
    return opts

class cope_():
    def __init__(self,opts):
        self.srcDir=opts.srcDir
        self.resDir=opts.resDir
        self.ratio=opts.ratio
        self.label2index = opts.label2index
        format = opts.format if isinstance(opts.format,list) else [opts.format] 
        self.format=format

    def _mkdir(self,D):
        os.makedirs(D,exist_ok=True)
    
    def _rename(self,n,pth,num):
        l =len(str(num)) 
        label_pool = pth.split(os.path.sep)
        labelA =  self.label2index['1'][label_pool[-4]]
        labelB =  self.label2index['2'][label_pool[-3]]
        labelC =  self.label2index['3'][label_pool[-2]]
        
        return f"{n:0{l}d}_{labelA}_{labelB}_{labelC}_.jpg" 
    def _loop(self,clsDir):
        imgPth_lst=[]
        for fmt in self.format: 
            imgPth_lst += list(pathlib.Path(clsDir).glob(f"*{fmt}"))
        num = len(imgPth_lst)
        random.shuffle(imgPth_lst)
        train_lst = imgPth_lst[:round(num*(1-self.ratio*2))]
        val_lst = imgPth_lst[round(num*(1-self.ratio*2)):round(num*(1-self.ratio))]
        test_lst = imgPth_lst[round(num*(1-self.ratio)):]
        Clsdirs_dct = {"train":train_lst,"val":val_lst,"test":test_lst} 
        for mode,dir_lst in tqdm.tqdm(Clsdirs_dct.items()):
            resDir = os.path.join(self.resDir,mode)
            self._mkdir(resDir)
            n = 0
            for pth in tqdm.tqdm(dir_lst,desc=f"{mode} "):
                pth = str(pth)
                new_pth = os.path.join(resDir,self._rename(n,pth,num))
                shutil.copyfile(pth,new_pth)
                n+=1
        show_lg(f"{clsDir}  over...\n",'')
    def run(self):
        clsDirs_lst = []
        for k1 in self.label2index["1"].keys():
            for k2 in self.label2index["2"].keys():
                for k3 in self.label2index["3"].keys():
                    clsDirs_lst.append(os.path.join(self.srcDir,k1,k2,k3))
        mps_lst= []
        for clsDir in clsDirs_lst:
            mps_lst.append(Process(target=self._loop,args=(clsDir,)))
        for mps in mps_lst:
            mps.start()
        for mps in mps_lst:
            mps.join()
    def get_info(self):
        show_lg('label',self.label2index)
        for mode in ['train','val','test']:
            show_lg(mode,len(list(pathlib.Path(os.path.join(self.resDir,mode)).glob("**/*.*"))))
if __name__ == "__main__":
    opts = parse_opt()
    func = cope_(opts) 
    func.run()
    show_lg("now: ",dt.datetime.now().strftime('%Y %m %d-%H:%M:%S'))
    func.get_info()