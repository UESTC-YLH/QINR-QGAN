from scipy.linalg import sqrtm
import numpy as np
import torch


def calculate_fid(act1, act2):
    act1 = act1.detach().cpu().numpy().reshape([1, 784])
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# def computer_KL():
#     computer_KL = torch.nn.functional.kl_div()
#     return computer_KL

def BoundarySeekingLoss(gloss):
    dx = torch.sigmoid(gloss)
    return 0.5 * torch.mean((torch.log(dx) - torch.log(1.0 - dx)) ** 2)

def calculate_cos(v1, v2):
    v1 = v1.detach().cpu().numpy().reshape(1, 784)
    v2 = v2.detach().cpu().numpy().reshape(1, 784)
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res