import torch.nn as nn 
import torch
import math


class EntropyDistorsionLoss(nn.Module):
    """
    Rate-distorsion loss based on hemp formulation of the entropy
    """

    def __init__(self, lmbda = 1e-2, mode = "factorized"):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.type == "icme"
        self.mode = mode





    def forward(self, output, target):

        N, _, H, W = target.size() 
  
        bs, ch, w,h  = output["likelihoods"]["y"].shape  #192,8192
        like_c = ch # 192
        like_dim = bs*w*h  #8192
        out = {}
        num_pixels = N * H * W
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        out["bpp_loss"] =   sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())    
        out["entropy"] =  -torch.sum(output["probability"]*(torch.log(output["probability"])/math.log(2)))*like_dim/(num_pixels)
        if self.mode == "factorized":
            out["bpp_loss"] =   sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
            out["loss"] =  self.lmbda * 255**2 * out["mse_loss"] + out["entropy"] 
            out["bpp_gauss"] = out["bpp_loss"]
            out["bpp_hype"] = out["bpp_loss"]
        else:
            
            bpp_loss_hype = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"]["z"])   
            bpp_loss_gauss = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"]["y"]) 
            out["bpp_loss"] =  bpp_loss_hype + bpp_loss_gauss
            out["loss"] =  self.lmbda * 255**2 * out["mse_loss"]   + out["entropy"] + bpp_loss_gauss
            out["bpp_gauss"] = bpp_loss_gauss
            out["bpp_hype"] = bpp_loss_hype

        return out


