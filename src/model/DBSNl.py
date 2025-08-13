import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse
import torch.nn.functional as F

from . import regist_model
from einops import rearrange
import numbers

from ddf import DDFPack



@regist_model
class DBSNl(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        super().__init__()
        self.net1=DBSN(in_ch, out_ch, base_ch, num_module)
        self.net2 = DBSN_(in_ch, out_ch, base_ch, num_module)
        self.net3=DBSN_1(in_ch, out_ch, base_ch, num_module)
        self.net4 =DBSN_x(in_ch, out_ch, base_ch, num_module)

    def forward(self, LL, LH,HL,HH):

        L_denoised = self.net1(LL)

        LH_denoised = self.net2(LH)  ###LH提取是竖边
        HL_denoised = self.net3(HL)  ###HL提取是横边
        HH_denoised = self.net4(HH)


        return L_denoised, LH_denoised,HL_denoised, HH_denoised

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DBSN(nn.Module):   #####DBSNl
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

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)


        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    # def _initialize_weights(self):
    #     # Liyong version
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):   #---------------------------------------------------------------------------------------->修改1-4
        return self.body(x)

        # conv_output = self.body[0](x)
        # conv_output = self.body[1](conv_output)
        # combined_output = conv_output + x
        # for layer in self.body[2:]:
        #     combined_output = layer(combined_output)
        # return combined_output









class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [nn.ReLU(inplace=True)]
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
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask

        # print("Current Weights Before Masking:")
        # print(self.weight.data)

        return super().forward(x)


##########################################增加的#################################################


########################################
########## 。。。。。
# class CentralMaskedConv2d_1(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         dis = kH // 2
#         for i in range(kH):
#             for j in range(kW):
#                 if j == dis:
#                     self.mask[:, :, i, j] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super().forward(x)
#
# class CentralMaskedConv2d_(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         dis = kH // 2
#         for i in range(kH):
#             for j in range(kW):
#                 if i == dis:
#                     self.mask[:, :, i, j] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super().forward(x)
#######。。。。。

class CentralMaskedConv2d_x(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        dis = kH // 2
        for i in range(kH):
            for j in range(kW):
                if i == dis:
                    self.mask[:, :, i, j] = 0
                if j == dis:
                    self.mask[:, :, i, j] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

#######。。。。。。。
# class DC_branchl_1(nn.Module):
#     def __init__(self, stride, in_ch, num_module):
#         super().__init__()
#
#         ly = []
#         ly += [CentralMaskedConv2d_1(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#
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
# class DC_branchl_(nn.Module):
#     def __init__(self, stride, in_ch, num_module):
#         super().__init__()
#
#         ly = []
#         ly += [CentralMaskedConv2d_(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#
#
#         ly += [DCl(stride, in_ch) for _ in range(num_module )]
#
#         ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
#         ly += [nn.ReLU(inplace=True)]
#
#         self.body = nn.Sequential(*ly)
#
#     def forward(self, x):
#         return self.body(x)
######。。。。

class DC_branchl_x(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d_x(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]


        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)








######。。。。。。。。。
# class DBSN_1(nn.Module):   #####DBSNl
#     '''
#     Dilated Blind-Spot Network (cutomized light version)
#
#     self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
#     and several modificaions are included.
#     see our supple for more details.
#     '''
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
#         assert base_ch%2 == 0, "base channel should be divided with 2"
#
#         ly = []
#         ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         self.head = nn.Sequential(*ly)
#
#         self.branch1 = DC_branchl_1(2, base_ch, num_module)
#         self.branch2 = DC_branchl_1(3, base_ch, num_module)
#
#
#         ly = []
#         ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
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
# class DBSN_(nn.Module):   #####DBSNl
#     '''
#     Dilated Blind-Spot Network (cutomized light version)
#
#     self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
#     and several modificaions are included.
#     see our supple for more details.
#     '''
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
#         assert base_ch%2 == 0, "base channel should be divided with 2"
#
#         ly = []
#         ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         self.head = nn.Sequential(*ly)
#
#         self.branch1 = DC_branchl_(2, base_ch, num_module)
#         self.branch2 = DC_branchl_(3, base_ch, num_module)
#
#
#         ly = []
#         ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
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
# ######。。。。。。


class DBSN_x(nn.Module):   #####DBSNl
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

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl_x(2, base_ch, num_module)
        self.branch2 = DC_branchl_x(3, base_ch, num_module)


        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)


################################## 修改class DBSN_(nn.Module)和class DBSN_1(nn.Module)###########


class CentralMaskedConv2d_1(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        dis = kH // 2
        for i in range(kH):
            for j in range(kW):
                if j == dis:
                    self.mask[:, :, i, j] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)



class CentralMaskedConv2d_(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        dis = kH // 2
        for i in range(kH):
            for j in range(kW):
                if i == dis:
                    self.mask[:, :, i, j] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)




class DCl_1(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=(1,3), stride=1, padding=(0,stride), dilation=(1,stride)) ]  #kernel_size=3, padding=stride, dilation=stride
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class DCl_(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=(3,1), stride=1, padding=(stride,0), dilation=(stride,1)) ]#kernel_size=3, stride=1, padding=stride, dilation=stride)
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)




class DC_branchl_1(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d_1(in_ch, in_ch, kernel_size=(1,2 * stride - 1), stride=1, padding=(0,stride - 1))]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl_1(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)




class DC_branchl_(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d_(in_ch, in_ch, kernel_size=(2 * stride - 1,1 ), stride=1, padding=( stride-1,0))]  #kernel_size=2 * stride - 1, stride=1, padding=stride - 1
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl_(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)



class DBSN_1(nn.Module):   #####DBSNl
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

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl_1(2, base_ch, num_module)
        self.branch2 = DC_branchl_1(3, base_ch, num_module)


        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)


class DBSN_(nn.Module):   #####DBSNl
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

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl_(2, base_ch, num_module)
        self.branch2 = DC_branchl_(3, base_ch, num_module)


        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)


################################################  完毕  ##################################




# ############################################  DTB ########################################
# class DTB_branchl(nn.Module):
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
#         ly += [DCl(stride, in_ch) for _ in range(9-num_module)]
#         ly += [DTB(stride=stride, num_blocks=[num_module], dim=in_ch)]
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
#
#
#
#
#
# ####归一化
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')
#
# def to_4d(x,h,w):
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
#
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight
#
# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
#
#
#
# # ## Gated-Dconv Feed-Forward Network (GDFN)
# # class FeedForward_dilated(nn.Module):
# #     def __init__(self, dim, ffn_expansion_factor, bias, stride):
# #         super(FeedForward_dilated, self).__init__()
# #
# #         hidden_features = int(dim*ffn_expansion_factor)
# #
# #         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
# #
# #         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=stride, groups=hidden_features*2, bias=bias, dilation=stride)
# #
# #         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
# #
# #     def forward(self, x):
# #         x = self.project_in(x)
# #         x1, x2 = self.dwconv(x).chunk(2, dim=1)
# #         x = F.gelu(x1) * x2
# #         x = self.project_out(x)
# #         return x
#
# ######  FeedForward_dilated---->修改成下面部分
# class FeedForward_dilated(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias,stride):
#         super(FeedForward_dilated, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, dilation=stride, padding=stride, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, dilation=stride, padding=stride, bias=bias)
#         self.cov=nn.Conv2d(dim, dim, kernel_size=1)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x = F.gelu(x)
#         x = self.project_out(x)
#         x = F.gelu(x)
#         x=self.cov(x)
#         return x
#
#
#
#
#
#
# class Attention_dilated(nn.Module):
#     def __init__(self, dim, num_heads, bias, stride):
#         super(Attention_dilated, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=stride, groups=dim * 3, bias=bias, dilation=stride)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#
#
# class DTB_block(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, stride):
#         super(DTB_block, self).__init__()
#
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention_dilated(dim, num_heads, bias, stride=stride)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward_dilated(dim, ffn_expansion_factor, bias, stride=stride)
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#
#         return x
#
#
# class DTB(nn.Module):
#     def __init__(self,
#                  inp_channels=3,
#                  out_channels=3,
#                  dim=128,
#                  num_blocks=[4],
#                  heads=[2],
#                  ffn_expansion_factor=1,
#                  bias=False,
#                  LayerNorm_type='BiasFree',  ## Other option 'BiasFree'  ffn_expansion_factor=2.66
#                  stride=2
#                  ):
#
#         super(DTB, self).__init__()
#
#
#         self.encoder_level1 = nn.Sequential(*[
#             DTB_block(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                              LayerNorm_type=LayerNorm_type, stride=stride) for i in range(num_blocks[0])])
#         # self.output = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=stride, bias=bias, dilation=stride)
#
#
#     def forward(self, inp_img):
#
#         b, c, h, w = inp_img.shape
#
#         out_enc_level1 = self.encoder_level1(inp_img)
#         out_dec_level1 = out_enc_level1
#         # out_dec_level1 = self.output(out_dec_level1)
#
#         return out_dec_level1

#############################################  DTB 完毕   ###############################################


