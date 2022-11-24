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

        
        
        out["bpp_hype"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_gauss"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        
        
        """
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        """
        out["bpp_loss"] = out["bpp_hype"] + out["bpp_gauss"]
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out