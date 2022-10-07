import torchvision.models as models
import torch
import torch.nn as nn
import onnx

# 1-模型加载
base_model= models.resnet18(num_classes=2)
torch.save(base_model,'base_model.pt')

# 2-训练阶段加入前处理+后处理
"""
后处理直接在生成模型上加了上去，而不是在训练完的后行后面家的
"""
class res18c(nn.Module):
    def __init__(self):
        super().__init__()
        self.mb = models.resnet18(num_classes=2)
    def forward(self,x):
        x = torch.divide(x,255.)
        x = self.mb(x)
        return torch.max(x,1).values
data = torch.randn(4,3,224,224)
res18c()(data)
"""output
tensor([-0.0775, -0.0217, -0.0766, -0.0535], grad_fn=<MaxBackward0>)
"""
m = res18c().eval()  # 这里很关键，torch.max不支持反向求导
torch.onnx.export(m,data,'propost.onnx',input_names=['hello'],output_names=['world'])
om = onnx.load('propost.onnx')
onnx.checker.check_model(om)

# 3- 在生成模型后面追加一个后处理
"""
两种方式
- 1 回载模型参数，追加结构
- 2 构建由后处理的网络结构，回载模型参数  这个最不好搞得。
"""
# 这里只测试第一种方法
res18 = models.resnet18(num_classes=2)
res18.load_state_dict(torch.load('base_model.pt',map_location='cpu').state_dict())
"""output
<All keys matched successfully>
"""
class res18c(nn.Module):
    def __init__(self):
        super().__init__()
        self.mb = res18
    def forward(self,x):
        x = torch.divide(x,255.)
        x = self.mb(x)
        return torch.max(x,1).values
res18_cc = res18c()(data)
res18_cc 
"""output
tensor([-0.0337,  0.0856,  0.1071, -0.0927], grad_fn=<MaxBackward0>)
"""