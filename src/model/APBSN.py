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
# def tensor2np(t:torch.Tensor):
#     '''
#     transform torch Tensor to numpy having opencv image form.
#     RGB -> BGR
#     (c,h,w) -> (h,w,c)
#     '''
#     t = t.cpu().detach()
#
#     # gray
#     if len(t.shape) == 2:
#         return t.permute(1,2,0).numpy()
#     # RGB -> BGR
#     elif len(t.shape) == 3:
#         return np.flip(t.permute(1,2,0).numpy(), axis=2)
#     # image batch
#     elif len(t.shape) == 4:
#         return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
#     else:
#         raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))
#
#
# def save_img_numpy( dir_name: str, file_name: str, img: np.array, ext='png'):
#     file_dir_name = os.path.join(dir_name, '%s.%s' % (file_name, ext))
#     if np.shape(img)[2] == 1:
#         cv2.imwrite(file_dir_name, np.squeeze(img, 2))
#     else:
#         cv2.imwrite(file_dir_name, img.squeeze(0))
# def save_img_tensor( dir_name: str, file_name: str, img: torch.Tensor, ext='png'):
#     save_img_numpy(dir_name, file_name, tensor2np(img), ext)
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
        # denosied_LH = LH_bsn.unsqueeze(2)
        # H.append(denosied_LH)
        # denosied_HL = HL_bsn.unsqueeze(2)
        # H.append(denosied_HL)
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

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:,:,:h,:w]
        else:
            # YLL, YH = self.dwt(img_pd_bsn)
            # threshold = 5  # np.sqrt(2 * np.log(np.prod(img.shape[1:]))) # 通用阈值选择
            #
            # YH_thresh = []
            # for coeff in YH:
            #     # 对每个高频系数应用软阈值
            #     coeff_thresh = torch.sign(coeff) * torch.maximum(torch.abs(coeff) - threshold,
            #                                                      torch.tensor(0.0, device=coeff.device))
            #     YH_thresh.append(coeff_thresh)
            #
            # de_img = self.idwt((YLL, YH_thresh))

            #####################################旋转###########################################

            # tmp_input=img_pd_bsn.flip(b + 1).transpose(b + 1, b + 2)
            #
            #
            # # do dwt
            # YLL, YH = self.dwt(tmp_input)
            # p = 2
            #
            # # else: self.pd_pad=!0
            # YLL = F.pad(YLL, (p, p, p, p), mode='reflect')
            # LH = F.pad(YH[0][:, :, 0, :, :], (p, p, p, p), mode='reflect')
            # HL = F.pad(YH[0][:, :, 1, :, :], (p, p, p, p), mode='reflect')
            # HH = F.pad(YH[0][:, :, 2, :, :], (p, p, p, p), mode='reflect')
            #
            # YLL_denoised, LH_denoised, HL_denoised, HH_denoised = self.bsn(YLL, LH, HL, HH)
            #
            # YLL_bsn = YLL_denoised[:, :, p:-p, p:-p]
            # LH_bsn = LH_denoised[:, :, p:-p, p:-p]
            # HL_bsn = HL_denoised[:, :, p:-p, p:-p]
            # HH_bsn = HH_denoised[:, :, p:-p, p:-p]
            #
            # H = []
            # LH_denosied = LH_bsn.unsqueeze(2)
            # H.append(LH_denosied)
            # HL_denosied = HL_bsn.unsqueeze(2)
            # H.append(HL_denosied)
            # HH_denosied = HH_bsn.unsqueeze(2)
            # H.append(HH_denosied)
            #
            # concat = torch.cat(H, dim=2)
            # YH_denoied = [concat]
            #
            # # do idwt
            # denoised= self.idwt((YLL_bsn, YH_denoied))  # denoised
            #
            #
            #
            # res = denoised.flip(b + 2).transpose(b + 1, b + 2)
            #
            #



            # ############################------->在图片上做r3  start #############################################
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]

                p=2
                # tmp_input = torch.clone(img_pd_bsn).detach()

                ###########################  修改 ############################
                # do dwt
                YLL, YH = self.dwt(tmp_input)

                # else: self.pd_pad=!0
                YLL = F.pad(YLL, (p,p,p,p), mode='reflect')
                LH = F.pad(YH[0][:, :, 0, :, :],(p,p,p,p), mode='reflect')
                HL = F.pad(YH[0][:, :, 1, :, :], (p,p,p,p), mode='reflect')
                HH = F.pad(YH[0][:, :, 2, :, :], (p,p,p,p), mode='reflect')

                YLL_denoised, LH_denoised,HL_denoised,HH_denoised = self.bsn(YLL, LH,HL,HH)




                YLL_bsn = YLL_denoised[:, :, p:-p, p:-p]
                LH_bsn = LH_denoised[:, :, p:-p, p:-p]
                HL_bsn = HL_denoised[:, :, p:-p, p:-p]
                HH_bsn = HH_denoised[:, :, p:-p, p:-p]

                H = []
                LH_denosied = LH_bsn.unsqueeze(2)
                H.append(LH_denosied)
                HL_denosied = HL_bsn.unsqueeze(2)
                H.append( HL_denosied)
                HH_denosied = HH_bsn.unsqueeze(2)
                H.append(HH_denosied)

                concat = torch.cat(H, dim=2)
                YH_denoied = [concat]

                # do idwt
                denoised[..., t]= self.idwt((YLL_bsn, YH_denoied))#denoised
            ###############################################在小波变换后做R3################################
            # p = self.pd_pad
            # # do dwt
            # YLL_x, YH_x = self.dwt(x)
            # cat_xH=torch.cat([YLL_x,YH_x[0][:, :, 0, :, :], YH_x[0][:, :, 1, :, :],YH_x[0][:, :, 2, :, :]], dim=0)
            #
            #
            # denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            # for t in range(self.R3_T):
            #     indice = torch.rand_like(cat_xH[0])
            #     mask = indice < self.R3_p
            #     mask = mask.unsqueeze(0)
            #     mask = mask.repeat(4, 1, 1, 1)
            #     tmp_concat = torch.clone(concat).detach()
            #     tmp_YLL=torch.clone(YLL_pd_bsn).detach()
            #     cat_bsnH = torch.cat([tmp_YLL,tmp_concat[:, :, 0, :, :], tmp_concat[:, :, 1, :, :], tmp_concat[:, :, 2, :, :]],
            #                          dim=0)
            #     cat_bsnH[mask]=cat_xH[mask]
            #
            #
            #     # else: self.pd_pad=!0
            #     YLL = F.pad(cat_bsnH[0].unsqueeze(0), (p, p, p, p), mode='reflect')
            #     LH = F.pad(cat_bsnH[1].unsqueeze(0),(p,p,p,p), mode='reflect')
            #     HL = F.pad(cat_bsnH[2].unsqueeze(0), (p,p,p,p), mode='reflect')
            #     HH = F.pad(cat_bsnH[3].unsqueeze(0), (p,p,p,p), mode='reflect')
            #
            #     YLL_denoised, LH_denoised, HL_denoised, HH_denoised = self.bsn(YLL, LH, HL, HH)
            #
            #     YLL_bsn = YLL_denoised[:, :, p:-p, p:-p]
            #     LH_bsn = LH_denoised[:, :, p:-p, p:-p]
            #     HL_bsn = HL_denoised[:, :, p:-p, p:-p]
            #     HH_bsn = HH_denoised[:, :, p:-p, p:-p]
            #
            #     H = []
            #     denosied_LH = LH_bsn.unsqueeze(2)
            #     H.append(denosied_LH)
            #     denosied_HL = HL_bsn.unsqueeze(2)
            #     H.append( denosied_HL)
            #     denosied_HH = HH_bsn.unsqueeze(2)
            #     H.append(denosied_HH)
            #
            #     concaton = torch.cat(H, dim=2)
            #     YH_denoied = [concaton]
            #
            #     # do idwt
            #     denoised[..., t] = self.idwt((YLL_bsn, YH_denoied))
            return torch.mean(denoised, dim=-1)  #denoised


##############################   小波变换 低频和高频分开训练两个网络  #################################################################
# class APBSN(nn.Module):
#     '''
#     Asymmetric PD Blind-Spot Network (AP-BSN)
#     '''
#     def __init__(self, pd_al=4, pd_ah=2,pd_bl=2, pd_bh=1, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
#                     bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
#         '''
#         Args:
#             pd_a           : 'PD stride factor' during training
#             pd_b           : 'PD stride factor' during inference
#             pd_pad         : pad size between sub-images by PD process
#             R3             : flag of 'Random Replacing Refinement'
#             R3_T           : number of masks for R3
#             R3_p           : probability of R3
#             bsn            : blind-spot network type
#             in_ch          : number of input image channel
#             bsn_base_ch    : number of bsn base channel
#             bsn_num_module : number of module
#         '''
#         super().__init__()
#
#         # network hyper-parameters
#         self.pd_al    = pd_al
#         self.pd_ah = pd_ah
#         self.pd_bl    = pd_bl
#         self.pd_bh = pd_bh
#         self.pd_pad  = pd_pad
#         self.R3      = R3
#         self.R3_T    = R3_T
#         self.R3_p    = R3_p
#
#         # define network
#         if bsn == 'DBSNl':
#             self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
#         else:
#             raise NotImplementedError('bsn %s is not implemented'%bsn)
#
#         self.dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()
#         self.idwt = DWTInverse(wave='haar', mode='zero').cuda()
#
#
#     def forward(self, img, pd_al=None,pd_ah=None):
#         '''
#         Foward function includes sequence of PD, BSN and inverse PD processes.
#         Note that denoise() function is used during inference time (for differenct pd factor and R3).
#         '''
#
#
#         #################################################
#         # default pd factor is training factor (a)
#         if pd_al is None: pd_al = self.pd_al
#         if pd_ah is None: pd_ah = self.pd_ah
#
#         # do dwt
#         YLL, YH = self.dwt(img)
#
#
#         # do PD
#         if pd_al > 1:
#             pd_YLL = pixel_shuffle_down_sampling(YLL, f=pd_al, pad=self.pd_pad)
#         else:
#             p = self.pd_pad
#             pd_YLL= F.pad(YLL, (p, p, p, p))
#
#         if pd_ah > 1:
#             pd_LH = pixel_shuffle_down_sampling(YH[0][:, :, 0, :, :],f=pd_ah, pad=self.pd_pad)
#             pd_HL = pixel_shuffle_down_sampling(YH[0][:, :, 1, :, :], f=pd_ah, pad=self.pd_pad)
#             pd_HH = pixel_shuffle_down_sampling(YH[0][:, :, 2, :, :], f=pd_ah, pad=self.pd_pad)
#
#
#         else:
#             p = self.pd_pad
#             pd_LH = F.pad(YH[0][:, :, 0, :, :], (p, p, p, p))
#             pd_HL = F.pad(YH[0][:, :, 1, :, :], (p, p, p, p))
#             pd_HH = F.pad(YH[0][:, :, 2, :, :],(p, p, p, p))
#
#
#         cat = torch.cat([pd_LH, pd_HL, pd_HH], dim=0)
#
#         # forward blind-spot network
#         pd_YLL_denoised, pd_denoised = self.bsn(pd_YLL, cat)
#
#
#
#         # do inverse PD
#         if pd_al > 1:
#             YLL_pd_bsn = pixel_shuffle_up_sampling(pd_YLL_denoised, f=pd_al, pad=self.pd_pad)
#         else:
#             p = self.pd_pad
#             YLL_pd_bsn = pd_YLL_denoised[:, :, p:-p, p:-p]
#
#         height = pd_denoised.shape[2]
#         width = pd_denoised.shape[3]
#
#         pd_denoised = pd_denoised.reshape(3, -1, 3, height, width)
#         pd_denosied_LH = pd_denoised[0, ...]
#         pd_denosied_HL = pd_denoised[1, ...]
#         pd_denosied_HH = pd_denoised[2, ...]
#
#         if pd_ah > 1:
#             LH_pd_bsn = pixel_shuffle_up_sampling(pd_denosied_LH, f=pd_ah, pad=self.pd_pad)
#             HL_pd_bsn = pixel_shuffle_up_sampling(pd_denosied_HL,f=pd_ah, pad=self.pd_pad)
#             HH_pd_bsn = pixel_shuffle_up_sampling(pd_denosied_HH, f=pd_ah, pad=self.pd_pad)
#
#         else:
#             p = self.pd_pad
#             LH_pd_bsn = pd_denosied_LH[:, :, p:-p, p:-p]
#             HL_pd_bsn = pd_denosied_HL[:, :, p:-p, p:-p]
#             HH_pd_bsn = pd_denosied_HH[:, :, p:-p, p:-p]
#
#         H=[]
#         denosied_LH = LH_pd_bsn.unsqueeze(2)
#         H.append(denosied_LH)
#         denosied_HL = HL_pd_bsn.unsqueeze(2)
#         H.append(denosied_HL)
#         denosied_HH = HH_pd_bsn.unsqueeze(2)
#         H.append(denosied_HH)
#
#         concat = torch.cat(H, dim=2)
#         YH_denoied = [concat]
#
#
#         # do idwt
#         img_pd_bsn = self.idwt((YLL_pd_bsn, YH_denoied))
#
#
#         ################################测试############################
#         return img_pd_bsn
#
#
#
#
#      ###########################################################################################
#
#     def denoise(self, x):
#         '''
#         Denoising process for inference.
#         '''
#         b,c,h,w = x.shape
#
#         # pad images for PD process
#         # if h % self.pd_b != 0:
#         #     x = F.pad(x, (0, 0, 0, self.pd_b - h%self.pd_b), mode='constant', value=0)
#         # if w % self.pd_b != 0:
#         #     x = F.pad(x, (0, self.pd_b - w%self.pd_b, 0, 0), mode='constant', value=0)
#
#         # forward PD-BSN process with inference pd factor
#         img_pd_bsn = self.forward(img=x, pd_al=self.pd_bl,pd_ah=self.pd_bh)
#
#         # Random Replacing Refinement
#         if not self.R3:
#             ''' Directly return the result (w/o R3) '''
#             return img_pd_bsn[:,:,:h,:w]
#         else:
#             # ############################------->在图片上做r3  start #############################################
#             # denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
#             # for t in range(self.R3_T):
#             #     indice = torch.rand_like(x)
#             #     mask = indice < self.R3_p
#             #
#             #     tmp_input = torch.clone(img_pd_bsn).detach()
#             #     tmp_input[mask] = x[mask]
#             #     p = self.pd_pad
#             #     ###########################  修改 ############################
#             #     # do dwt
#             #     YLL, YH = self.dwt(tmp_input)
#             #
#             #     # else: self.pd_pad=!0
#             #     YLL = F.pad(YLL, (p,p,p,p), mode='reflect')
#             #     LH = F.pad(YH[0][:, :, 0, :, :],(p,p,p,p), mode='reflect')
#             #     HL = F.pad(YH[0][:, :, 1, :, :], (p,p,p,p), mode='reflect')
#             #     HH = F.pad(YH[0][:, :, 2, :, :], (p,p,p,p), mode='reflect')
#             #     cat = torch.cat([LH, HL, HH], dim=0)
#             #
#             #     YLL_denoised, H_denoised = self.bsn(YLL, cat)
#             #
#             #     YLL_bsn = YLL_denoised[:, :, p:-p, p:-p]
#             #
#             #     height = H_denoised.shape[2]
#             #     width = H_denoised.shape[3]
#             #
#             #     H_denoised = H_denoised.reshape(3, -1, 3, height, width)
#             #     denosied_LH = H_denoised[0, ...]
#             #     denosied_HL = H_denoised[1, ...]
#             #     denosied_HH =H_denoised[2, ...]
#             #
#             #     LH_bsn = denosied_LH[:, :, p:-p, p:-p]
#             #     HL_bsn = denosied_HL[:, :, p:-p, p:-p]
#             #     HH_bsn = denosied_HH[:, :, p:-p, p:-p]
#             #
#             #     H = []
#             #     LH_denosied = LH_bsn.unsqueeze(2)
#             #     H.append(LH_denosied)
#             #     HL_denosied = HL_bsn.unsqueeze(2)
#             #     H.append( HL_denosied)
#             #     HH_denosied = HH_bsn.unsqueeze(2)
#             #     H.append(HH_denosied)
#             #
#             #     concat = torch.cat(H, dim=2)
#             #     YH_denoied = [concat]
#             #
#             #     # do idwt
#             #     denoised[..., t] = self.idwt((YLL_bsn, YH_denoied))
#             #     ####################------------>end   #####################################
#             ##########################------------> 在小波变换后做r3  ##############################
#             # denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
#             # for t in range(self.R3_T):
#             #     indice = torch.rand_like(x)
#             #     mask = indice < self.R3_p
#             #
#             #     tmp_input = torch.clone(img_pd_bsn).detach()
#             #     tmp_input[mask] = x[mask]
#             #     p = self.pd_pad
#             ################################  修改 ############################
#             p = self.pd_pad
#             # do dwt
#             YLL_x, YH_x = self.dwt(x)
#             cat_xLH=torch.cat([YLL_x, YH_x[0][:, :, 0, :, :], YH_x[0][:, :, 1, :, :],YH_x[0][:, :, 2, :, :]], dim=0)
#
#             tmp_input = torch.clone(img_pd_bsn).detach()
#             YLL_bsn, YH_bsn = self.dwt(tmp_input)
#             cat_bsnLH = torch.cat([YLL_bsn,YH_bsn[0][:, :, 0, :, :], YH_bsn[0][:, :, 1, :, :], YH_bsn[0][:, :, 2, :, :]], dim=0)
#             h=YLL_bsn.squeeze(0)-cat_bsnLH[0]
#
#
#             denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
#             for t in range(self.R3_T):
#                 indice = torch.rand_like(cat_xLH)
#                 mask = indice < self.R3_p
#                 cat_bsnLH[mask]=cat_xLH[mask]
#
#
#                 # else: self.pd_pad=!0
#                 YLL = F.pad(cat_bsnLH[0].unsqueeze(0), (p,p,p,p), mode='reflect')
#                 LH = F.pad(cat_bsnLH[1].unsqueeze(0),(p,p,p,p), mode='reflect')
#                 HL = F.pad(cat_bsnLH[2].unsqueeze(0), (p,p,p,p), mode='reflect')
#                 HH = F.pad(cat_bsnLH[3].unsqueeze(0), (p,p,p,p), mode='reflect')
#                 cat = torch.cat([LH, HL, HH], dim=0)
#
#                 YLL_denoised, H_denoised = self.bsn(YLL, cat)
#
#                 YLL_bsn = YLL_denoised[:, :, p:-p, p:-p]
#
#                 height = H_denoised.shape[2]
#                 width = H_denoised.shape[3]
#
#                 H_denoised = H_denoised.reshape(3, -1, 3, height, width)
#                 denosied_LH = H_denoised[0, ...]
#                 denosied_HL = H_denoised[1, ...]
#                 denosied_HH =H_denoised[2, ...]
#
#                 LH_bsn = denosied_LH[:, :, p:-p, p:-p]
#                 HL_bsn = denosied_HL[:, :, p:-p, p:-p]
#                 HH_bsn = denosied_HH[:, :, p:-p, p:-p]
#
#                 H = []
#                 LH_denosied = LH_bsn.unsqueeze(2)
#                 H.append(LH_denosied)
#                 HL_denosied = HL_bsn.unsqueeze(2)
#                 H.append( HL_denosied)
#                 HH_denosied = HH_bsn.unsqueeze(2)
#                 H.append(HH_denosied)
#
#                 concat = torch.cat(H, dim=2)
#                 YH_denoied = [concat]
#
#                 # do idwt
#                 denoised[..., t] = self.idwt((YLL_bsn, YH_denoied))
#
#
#             return torch.mean(denoised, dim=-1)
#
#         '''
#         elif self.R3 == 'PD-refinement':
#             s = 2
#             denoised = torch.empty(*(x.shape), s**2, device=x.device)
#             for i in range(s):
#                 for j in range(s):
#                     tmp_input = torch.clone(x_mean).detach()
#                     tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
#                     p = self.pd_pad
#                     tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
#                     if self.pd_pad == 0:
#                         denoised[..., i*s+j] = self.bsn(tmp_input)
#                     else:
#                         denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
#             return_denoised = torch.mean(denoised, dim=-1)
#         else:
#             raise RuntimeError('post-processing type not supported')
#         '''

############################################### 原始代码   #####################################################################
# class APBSN(nn.Module):
#     '''
#     Asymmetric PD Blind-Spot Network (AP-BSN)
#     '''
#
#     def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
#                  bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
#         '''
#         Args:
#             pd_a           : 'PD stride factor' during training
#             pd_b           : 'PD stride factor' during inference
#             pd_pad         : pad size between sub-images by PD process
#             R3             : flag of 'Random Replacing Refinement'
#             R3_T           : number of masks for R3
#             R3_p           : probability of R3
#             bsn            : blind-spot network type
#             in_ch          : number of input image channel
#             bsn_base_ch    : number of bsn base channel
#             bsn_num_module : number of module
#         '''
#         super().__init__()
#
#         # network hyper-parameters
#         self.pd_a = pd_a
#         self.pd_b = pd_b
#         self.pd_pad = pd_pad
#         self.R3 = R3
#         self.R3_T = R3_T
#         self.R3_p = R3_p
#
#         # define network
#         if bsn == 'DBSNl':
#             self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
#         else:
#             raise NotImplementedError('bsn %s is not implemented' % bsn)
#
#     def forward(self, img, pd=None):
#         '''
#         Foward function includes sequence of PD, BSN and inverse PD processes.
#         Note that denoise() function is used during inference time (for differenct pd factor and R3).
#         '''
#         # default pd factor is training factor (a)
#         if pd is None: pd = self.pd_a
#
#         # do PD
#         if pd > 1:
#             pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
#         else:
#             p = self.pd_pad
#             pd_img = F.pad(img, (p, p, p, p))
#
#         # forward blind-spot network
#         pd_img_denoised = self.bsn(pd_img)  #####‘’‘  #######修改’‘’  ###############pd_img----img
#
#         # # do inverse PD
#         if pd > 1:
#             img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
#         else:
#             p = self.pd_pad
#             img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]
#
#         return img_pd_bsn  #####‘’‘  #######修改’‘’  ###############pd_img----img
#
#     def denoise(self, x):
#         '''
#         Denoising process for inference.
#         '''
#         b, c, h, w = x.shape
#
#         # pad images for PD process
#         if h % self.pd_b != 0:
#             x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
#         if w % self.pd_b != 0:
#             x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
#
#         # forward PD-BSN process with inference pd factor
#         img_pd_bsn = self.forward(img=x, pd=self.pd_b)
#
#         # Random Replacing Refinement
#         if not self.R3:
#             ''' Directly return the result (w/o R3) '''
#             return img_pd_bsn[:, :, :h, :w]
#         else:
#             denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
#             for t in range(self.R3_T):
#                 indice = torch.rand_like(x)
#                 mask = indice < self.R3_p
#
#                 tmp_input = torch.clone(img_pd_bsn).detach()
#                 tmp_input[mask] = x[mask]
#                 p = self.pd_pad
#                 tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
#                 if self.pd_pad == 0:
#                     denoised[..., t] = self.bsn(tmp_input)
#                 else:
#                     denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
#
#             return torch.mean(denoised, dim=-1)





