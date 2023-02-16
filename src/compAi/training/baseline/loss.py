import torch.nn as nn 
import torch
import math



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        #bpp_loss_hype = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"]["z"])   
        #bpp_loss_gauss = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"]["y"]) 
        #out["bpp_loss"] =  bpp_loss_hype + bpp_loss_gauss
        #out["bpp_gauss"] = bpp_loss_gauss
        #out["bpp_hype"] = bpp_loss_hype

        
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["bpp_gauss"] = out["bpp_loss"] 
        out["bpp_hype"] = out["bpp_loss"] 
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out