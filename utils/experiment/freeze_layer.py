import torchvision.models as m
import torch

def layer_some_layers(model,flag):
    ...
def func_freeze1(net):
    freeze  = [4]
    freeze = [f'layer{x}.' for x in (freeze if len(freeze) > 1 else range(1,freeze[0]+1))]  # layers to freeze
    print(freeze)
    for k, v in net.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            # print(f'freezing {k}')
            v.requires_grad = False


def func_freeze2(net):
    freeze_not  = ['fc.','layer4.']
    print(freeze_not)
    for k, v in net.named_parameters():
        v.requires_grad = False  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # print(any(x not in k for x in freeze_not))
        if any(x in k for x in freeze_not):
            print(f'freezing {k}')
            v.requires_grad = True

if __name__ == "__main__":
    
    net = m.resnet34()
    net.eval()
    with torch.no_grad():
        for n,v in net.named_parameters():
            print(n,'    ',v.requires_grad)

    # func_freeze1(net)
    # func_freeze2(net)
    # print('-----------------------')
    # for k, v in net.named_parameters():
    #     if v.requires_grad:
    #         print(f'{k}  grad is True')
    del net.fc
    print(net)
    net = torch.nn.Sequential(net,torch.nn.Conv2d(512,10,1))
    print(net)