import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling,wavelets_split,wavelets_merge
from . import regist_model
from .DBSNl import DBSNl
import time
import matplotlib.pyplot as plt
import numpy as np

#
# import numpy as np
import os ,cv2

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).astype('float32')

def read_img(filepath):
    img = cv2.imread(filepath)  # 读的图片格式是BGR
    img = img[:, :, ::-1] #/ 255.0  # 将BGR格式转换成RGB格式，再归一化
    img = np.array(img).astype('float32')  # dtype('float64')转换成dtype('float32')
    return img


@regist_model
class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''
    def __init__(self,pd_al=4, pd_ah=2,pd_bl=2, pd_bh=1, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,  #########修改-------------------
                    bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_al    = pd_al
        self.pd_ah = pd_ah
        self.pd_bl    = pd_bl
        self.pd_bh = pd_bh
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p


        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)

        self.dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()  ###'haar' 'zero'   'db3' 'symmetric'               'bior2.4'
        self.idwt = DWTInverse(wave='haar', mode='zero').cuda()      ####'haar'  'zero'                         'bior2.4'


    def forward(self, img, pd_al=None,pd_ah=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        #################################################
        # default pd factor is training factor (a)
        if pd_al is None: pd_al = self.pd_al
        if pd_ah is None: pd_ah = self.pd_ah

        # do dwt
        YLL, YH = self.dwt(img)


        # do PD
        if pd_al > 1:#pd_al > 1  修改=====
            pd_YLL = pixel_shuffle_down_sampling(YLL, f=pd_al, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_YLL= F.pad(YLL, (p, p, p, p))

        if pd_ah > 1:
            # pd_LH = pixel_shuffle_down_sampling(YH[0][:, :, 0, :, :],f=pd_ah, pad=self.pd_pad)
            # pd_HL = pixel_shuffle_down_sampling(YH[0][:, :, 1, :, :], f=pd_ah, pad=self.pd_pad)
            pd_LH = wavelets_split(YH[0][:, :, 1, :, :],factor_h=pd_al,factor_w=pd_ah, pad=self.pd_pad)   #0--->1
            pd_HL = wavelets_split(YH[0][:, :, 0, :, :], factor_h=pd_ah,factor_w=pd_al, pad=self.pd_pad)    #1--->0

            pd_HH = pixel_shuffle_down_sampling(YH[0][:, :, 2, :, :], f=pd_ah, pad=self.pd_pad)

        else:
            p = self.pd_pad
            pd_LH = F.pad(YH[0][:, :, 0, :, :], (p, p, p, p))  #0--->1
            pd_HL = F.pad(YH[0][:, :, 1, :, :], (p, p, p, p))
            pd_HH = F.pad(YH[0][:, :, 2, :, :],(p, p, p, p))



        # forward blind-spot network
        pd_YLL_denoised, pd_LH_denoised , pd_HL_denoised, pd_HH_denoised= self.bsn(pd_YLL, pd_LH,pd_HL,pd_HH)  #L_denoised, LH_denoised,HL_denoised, HH_denoised


        # do inverse PD
        if pd_al > 1:   #pd_al > 1  修改=====
            YLL_pd_bsn = pixel_shuffle_up_sampling(pd_YLL_denoised, f=pd_al, pad=self.pd_pad)
        else:
            p = self.pd_pad
            YLL_pd_bsn = pd_YLL_denoised[:, :, p:-p, p:-p]

        if pd_ah > 1:
            # LH_bsn = pixel_shuffle_up_sampling(pd_LH_denoised, f=pd_ah, pad=self.pd_pad)
            # HL_bsn = pixel_shuffle_up_sampling(pd_HL_denoised,f=pd_ah, pad=self.pd_pad)
            LH_bsn = wavelets_merge(pd_LH_denoised, factor_h=pd_al,factor_w=pd_ah, pad=self.pd_pad)
            HL_bsn = wavelets_merge(pd_HL_denoised,factor_h=pd_ah,factor_w=pd_al, pad=self.pd_pad)



            HH_bsn = pixel_shuffle_up_sampling(pd_HH_denoised, f=pd_ah, pad=self.pd_pad)

        else:
            p = self.pd_pad
            LH_bsn = pd_LH_denoised[:, :, p:-p, p:-p]
            HL_bsn = pd_HL_denoised[:, :, p:-p, p:-p]
            HH_bsn = pd_HH_denoised[:, :, p:-p, p:-p]

        H=[]
        denosied_HL = HL_bsn.unsqueeze(2)
        H.append(denosied_HL)
        denosied_LH = LH_bsn.unsqueeze(2)
        H.append(denosied_LH)
        denosied_HH = HH_bsn.unsqueeze(2)
        H.append(denosied_HH)
        concat = torch.cat(H, dim=2)
        YH_denoied = [concat]


        # do idwt
        img_pd_bsn = self.idwt((YLL_pd_bsn, YH_denoied))#self.idwt((YLL_pd_bsn, YH_denoied))

        return img_pd_bsn  ###########修改,YLL_pd_bsn,concat



    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b,c,h,w = x.shape

        # pad images for PD process
        # if h % self.pd_b != 0:
        #     x = F.pad(x, (0, 0, 0, self.pd_b - h%self.pd_b), mode='constant', value=0)
        # if w % self.pd_b != 0:
        #     x = F.pad(x, (0, self.pd_b - w%self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor

        img_pd_bsn = self.forward(img=x, pd_al=self.pd_bl,pd_ah=self.pd_bh)#pd_ah####修改-------------------


        return img_pd_bsn[:,:,:h,:w]
