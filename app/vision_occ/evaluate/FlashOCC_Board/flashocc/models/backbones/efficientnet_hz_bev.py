import torch
import torch.nn as nn
from .efficientnet_no_se import EfficientNetHz
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet3d.models import BACKBONES

@BACKBONES.register_module()
class EfficientNetB0HZ(nn.Module):
    def __init__(self, pretrained=None, act_layer='relu6'):
        super(EfficientNetB0HZ, self).__init__()
        model = EfficientNetHz(
                num_features=1280,
                stem_size=32,
                fix_stem=True,
                act_layer=act_layer,
                norm_layer='BatchNorm2d')
        if pretrained is not None:
            print('pretrained backbone is from {}'.format(pretrained))
            state_dict = torch.load(pretrained, map_location='cpu')  # 读取权重文件
            model.load_state_dict(state_dict, strict=False)  # 加载权重到模型
        else:
            print('There is no pretrained model offered.')

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.conv_head = model.conv_head
        self.bn2 = model.bn2
        self.block_0 = model.blocks[0]
        self.block_1 = model.blocks[1]
        self.block_2 = model.blocks[2]
        self.block_3 = model.blocks[3]
        self.block_4 = model.blocks[4]
        self.block_5 = model.blocks[5]
        self.block_6 = model.blocks[6]
    
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x_4 = self.block_4(x) # 112, 16, 44
        x = self.block_5(x_4)
        x = self.block_6(x) # 320, 8, 22
        x = self.conv_head(x) # 1280, 8, 22
        x = self.bn2(x) # 1280, 8, 22
        return x_4, x

    