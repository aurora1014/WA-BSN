import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


######################加入TV损失################################
def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

# def TV_loss(x):
#     batch_size = x.size()[0]
#     h_x = x.size()[2]
#     w_x = x.size()[3]
#     count_h = _tensor_size(x[:, :, 1:, :])
#     count_w = _tensor_size(x[:, :, :, 1:])
#     h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#     w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#     return 0.05 * 2 * (h_tv / count_h + w_tv / count_w) / batch_size   #0.05


def TV_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]

    count_h = _tensor_size(x[:, :, 1:, :])  # 高度方向上的有效像素数量
    count_w = _tensor_size(x[:, :, :, 1:])  # 宽度方向上的有效像素数量

    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()  # 高度方向的总变差
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()  # 宽度方向的总变差

    return  (h_tv / count_h + w_tv / count_w) / batch_size   #0.05 * 2




###########################################################
eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['real_noisy'] #data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy) +0.35*TV_loss(output) #+0.35*TV_loss(output)    ########修改F.l1_loss(output, target_noisy)-----》F.l1_loss(output, target_noisy)+TV_loss(output)





@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)
