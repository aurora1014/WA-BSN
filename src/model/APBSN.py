
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model
from .DBSNl import DBSNl

import cv2,os,time
import numpy as np

def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))


def save_img_numpy( dir_name: str, file_name: str, img: np.array, ext='png'):
    file_dir_name = os.path.join(dir_name, '%s.%s' % (file_name, ext))
    if np.shape(img)[2] == 1:
        cv2.imwrite(file_dir_name, np.squeeze(img, 2))
    else:
        cv2.imwrite(file_dir_name, img.squeeze(0))
def save_img_tensor( dir_name: str, file_name: str, img: torch.Tensor, ext='png'):
    save_img_numpy(dir_name, file_name, tensor2np(img), ext)
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
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, 
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
        self.pd_a    = pd_a
        self.pd_b    = pd_b
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p
        
        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # img = cv2.imread('/home/lhy008/NIND/NIND_partiallyeatenbanana_ISO6400.png')
        # img  = torch.tensor(img , dtype=torch.float32).cuda()
        # img  = img [1200:2840, 1000:3400, :]
        #
        # img=img.unsqueeze(0)
        # img=img.permute(0, 3, 1, 2)


        # img = img[:, [2, 1, 0], :, :]
        # result_path = '/home/lhy008'
        # save_img_tensor(result_path, 'NIND_MuseeL-ram_ISO6400', img)





        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)

        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p,p,p,p))

        
        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)


        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:,:,p:-p,p:-p]
        #
        # result_path='/home/lhy008'
        # img_pd_bsn = img_pd_bsn[:, [2, 1, 0], :, :]
        # save_img_tensor(result_path, 'de_img5', img_pd_bsn)


        return img_pd_bsn

    def denoise(self, x):
        '''
        Denoising process for inference.

        '''
        b,c,h,w = x.shape

        # start_time = time.time()

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h%self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w%self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)

        # time_end = time.time()
        # time_sum = time_end - start_time

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:,:,:h,:w]
        else:

            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:,:,p:-p,p:-p]

            return torch.mean(denoised, dim=-1)
            
        '''
        elif self.R3 == 'PD-refinement':
            s = 2
            denoised = torch.empty(*(x.shape), s**2, device=x.device)
            for i in range(s):
                for j in range(s):
                    tmp_input = torch.clone(x_mean).detach()
                    tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
                    p = self.pd_pad
                    tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                    if self.pd_pad == 0:
                        denoised[..., i*s+j] = self.bsn(tmp_input)
                    else:
                        denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return_denoised = torch.mean(denoised, dim=-1)
        else:
            raise RuntimeError('post-processing type not supported')
        '''
            # s = 3
            # denoised = torch.empty(*(x.shape), s ** 2, device=x.device)
            # for i in range(s):
            #     for j in range(s):
            #         tmp_input = torch.clone(img_pd_bsn).detach()
            #         tmp_input[:, :, i::s, j::s] = x[:, :, i::s, j::s]
            #         p = self.pd_pad
            #         tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
            #         if self.pd_pad == 0:
            #             denoised[..., i * s + j] = self.bsn(tmp_input)
            #         else:
            #             denoised[..., i * s + j] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
            #             return_denoised = torch.mean(denoised, dim=-1)
            # return return_denoised