import torch
import torch.nn as nn
import torch.nn.functional as F

def  DistillKL (y_s, y_t):
    temp =4
    p_s = F.log_softmax(y_s/temp, dim=1)
    p_t = F.softmax(y_t/temp, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temp**2) / y_s.shape[0]
    return loss

"""def L2_loss(y_0, y_1):
    margine=torch.ones(2, 64, 13248)
    margine = margine.cuda()
    margine=margine*0.001
    y_1=y_1+margine
    loss = torch.pow(y_0 - y_1, 2)
    loss[loss != loss] = 0
    return loss.mean()
"""
def L2_loss(y_s, y_t):
    mask = (y_s != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_s - y_t, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()
    
"""def  DistillKL (y_s, y_t):
    p_s = F.softmax(y_s, dim=1)
    p_t = F.softmax(y_t, dim=1)
    #loss = F.kl_div(p_s, p_t, size_average=False)  / y_s.shape[0]
    loss=(p_t * torch.log(1e-8 + p_t / (p_s + 1e-8))) /y_s.shape[0]
    return loss.mean()
    #return loss
"""
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
