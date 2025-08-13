import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_model
#####ddf
import math
from einops import rearrange
import numbers
from ddf import DDFPack
from timm.models.resnet import Bottleneck, ResNet, _cfg

#
# @regist_model
# class DBSNl(nn.Module):
#     '''
#     Dilated Blind-Spot Network (cutomized light version)
#
#     self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
#     and several modificaions are included.
#     see our supple for more details.
#     '''
#
#     def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
#         '''
#         Args:
#             in_ch      : number of input channel
#             out_ch     : number of output channel
#             base_ch    : number of base channel
#             num_module : number of modules in the network
#         '''
#         super().__init__()
#
#         assert base_ch % 2 == 0, "base channel should be divided with 2"
#
#         ly = []
#         ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         self.head = nn.Sequential(*ly)
#
#         self.branch1 = DC_branchl(2, base_ch, num_module)
#         self.branch2 = DC_branchl(3, base_ch, num_module)
#
#         ly = []
#         ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
#         self.tail = nn.Sequential(*ly)
#
#     def forward(self, x):
#         x = self.head(x)
#
#         br1 = self.branch1(x)
#         br2 = self.branch2(x)
#
#         x = torch.cat([br1, br2], dim=1)
#
#         return self.tail(x)
#
#     def _initialize_weights(self):
#         # Liyong version
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
#
#
# class DC_branchl(nn.Module):
#     def __init__(self, stride, in_ch, num_module):
#         super().__init__()
#
#         ly = []
#         ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#
#         ly += [DCl(stride, in_ch) for _ in range(num_module)]
#
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#
#         self.body = nn.Sequential(*ly)
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class DCl(nn.Module):
#     def __init__(self, stride, in_ch):
#         super().__init__()
#
#         ly = []
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         self.body = nn.Sequential(*ly)
#
#     def forward(self, x):
#         return x + self.body(x)
#
#
# class CentralMaskedConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         self.mask[:, :, kH // 2, kH // 2] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super().forward(x)
#
#




@regist_model
class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        # ly += [Golbal_block(in_ch,stride,num_heads=2,ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias_LayerNorm') for _ in range(3)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)



class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [nn.ReLU(inplace=True)]
        ly += [DDFPack(in_ch, kernel_size=3, stride=1,dilation=stride,padding=stride, kernel_combine='mul')]  #'mul'
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)










########################################################
#####Golbal_block
class Golbal_block(nn.Module):
    def __init__(self, in_ch, stride,num_heads, ffn_expansion_factor, bias, LayerNorm_type ):
        super(Golbal_block, self).__init__()

        self.norm1 = LayerNorm(in_ch, LayerNorm_type)
        self.attn = Attention_dilated(in_ch, num_heads, bias, stride=stride)
        self.norm2 = LayerNorm(in_ch, LayerNorm_type)
        # ly = []
        # ly += [LayerNorm(in_ch, LayerNorm_type)]
        # ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        # ly += [ nn.ReLU(inplace=True)]
        # self.body = nn.Sequential(*ly)
        hidden_features = int(in_ch * ffn_expansion_factor)
        self.con1 =nn.Conv2d(in_ch, hidden_features, kernel_size=3, stride=1, padding=stride, dilation=stride,bias=bias)
        self.con2 = nn.Conv2d(hidden_features, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride,bias=bias)

    def forward(self, x):
        x1 = x + self.attn(self.norm1(x))
        x1 =self.con1(self.norm2(x1))
        x1=F.gelu(x1)
        x1=self.con2(x1)
        return x1





##########################################################
######Attention block
class Attention_dilated(nn.Module):
    def __init__(self, dim, num_heads, bias, stride):
        super(Attention_dilated, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=stride, groups=dim * 3, bias=bias, dilation=stride)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


####################################################################
######层归一化

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)








