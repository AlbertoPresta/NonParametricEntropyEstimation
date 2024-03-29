import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from compressai.ans import BufferedRansEncoder, RansDecoder
import warnings

from compressai.models.utils import conv, deconv



from compressai.layers import (
    GDN,
    MaskedConv2d,
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


from compAi.entropy_bottleneck.papers.ICME2023.entropy_bottleneck import   EntropyBottleneck, GaussianConditional

import time



import warnings
warnings.filterwarnings("ignore")



class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.
    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels = 192, extrema = 30, power = 1, delta = 1):
        super().__init__()
        
        
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels, extrema = extrema, power = power, delta = delta)


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m,   EntropyBottleneck)
        )
        return aux_loss

    def forward(self, *args):
        raise NotImplementedError()

    def update(self,  device = torch.device("cpu"),stat_pmf = None):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        updated = False
        for m in self.children():
            if not isinstance(m,  EntropyBottleneck):
                continue
            rv = m.update(device = device)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict = False):
        
        if "entropy_bottleneck.pmf_linear.weight" in state_dict.keys():
            del state_dict["entropy_bottleneck.pmf_linear.weight"]
            del state_dict["entropy_bottleneck.pmf_linear.bias"]
        if "entropy_bottleneck.target" in state_dict.keys():
            print("sono qua")
        del state_dict["entropy_bottleneck._offset"]
        del state_dict["entropy_bottleneck._quantized_cdf"] 
        del state_dict["entropy_bottleneck._cdf_length"] 
        
        
        super().load_state_dict(state_dict,strict = strict)



class FactorizedICME(CompressionModel):
    def __init__(self, N, M,extrema = 30,power = 1.0, **kwargs):
        super().__init__(entropy_bottleneck_channels = M,extrema = extrema, power = power, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M
        self.extrema = extrema   
        self.stat_pmf = None
        self.pmf = None





                
    def define_statistical_pmf(self, dataloader,device =torch.device("cuda"),  idx = 100):
        res = torch.zeros((self.M,2*self.extrema +1)).to(device)     
        start = time.time() 
        for dim in range(self.M):
            temp_res = torch.zeros((idx,2*self.extrema +1)).to(device)           
            with torch.no_grad():
                for i,d in enumerate(dataloader):
                    if i > idx - 1:
                        break
                    d = d.to(device)
                    out_enc = self.g_a(d) 
                    bs, ch, w,h = out_enc.shape
                    out_enc = out_enc.round().int() #these dshould be the latent space
                    out_enc = out_enc.reshape(bs,ch,w*h)
                    unique, val = torch.unique(out_enc[0,dim,:], return_counts = True)
                    dict_val =  dict(zip(unique.tolist(), val.tolist()))
                    cc = 0 
                    for j in range(-self.extrema, self.extrema + 1):
                        if j not in unique:
                            temp_res[i,cc]  = 0 
                            cc = cc + 1
                        else:
                            temp_res[i,cc] = dict_val[j]
                            cc = cc +1
                temp_res = torch.sum(temp_res, dim = 0)
                temp_res = temp_res.reshape(-1)
                temp_res = temp_res/torch.sum(temp_res)
                res[dim,:] = temp_res  
        print("time : ",time.time() -start )
        print("shape of statistical res is: ",res.shape)
        self.entropy_bottleneck.stat_pmf = res   
    
    def forward(self, x, training = False):
        y = self.g_a(x)
        y_hat, y_likelihoods, y_probability = self.entropy_bottleneck(y,training)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
            "probability": y_probability
        }

    
    def compress_during_training(self, x, device = torch.device("cuda")):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress_during_training(y, device = device)
        return {"strings": [y_strings], "shape": y.size()[-2:]}


    def decompress_during_training(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress_during_training(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

   
    
    def compress(self, x):
        y = self.g_a(x)        
        byte_stream, output_cdf = self.entropy_bottleneck.compress(y)
        return byte_stream, output_cdf, {"strings": byte_stream, "shape": y.size()[-2:]}

    def decompress(self, byte_stream, output_cdf):
        #assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(byte_stream, output_cdf)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
 

###########################################################################################################  
###########################################################################################################   
############################################# HYPERPRIOR ##################################################
###########################################################################################################
###########################################################################################################

 
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) 
    
class ICMEScaleHyperprior(CompressionModel):
    """
    """

    def __init__(self, N, M,extrema = 30, power = 1, delta = 1, **kwargs):
        super().__init__(entropy_bottleneck_channels = N,extrema = extrema, power = power, delta = delta, **kwargs)

        
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.extrema = extrema   
        self.stat_pmf = None
        self.delta = delta
        self.power = power



    def define_statistical_pmf(self, dataloader,device =torch.device("cuda"),  idx = 100):
        res = torch.zeros((self.N,2*self.extrema +1)).to(device)     
        start = time.time() 
        for dim in range(self.N):
            temp_res = torch.zeros((idx,2*self.extrema +1)).to(device)           
            with torch.no_grad():
                for i,d in enumerate(dataloader):
                    print(i)
                    if i > idx - 1:
                        break
                    d = d.to(device)
                    out_enc = self.g_a(d) 
                    bs, ch, w,h = out_enc.shape
                    out_enc = out_enc.round().int() #these dshould be the latent space
                    out_enc = out_enc.reshape(bs,ch,w*h)
                    unique, val = torch.unique(out_enc[0,dim,:], return_counts = True)
                    dict_val =  dict(zip(unique.tolist(), val.tolist()))
                    cc = 0 
                    for j in range(-self.extrema, self.extrema + 1):
                        if j not in unique:
                            temp_res[i,cc]  = 0 
                            cc = cc + 1
                        else:
                            temp_res[i,cc] = dict_val[j]
                            cc = cc +1
                temp_res = torch.sum(temp_res, dim = 0)
                temp_res = temp_res.reshape(-1)
                temp_res = temp_res/torch.sum(temp_res)
                res[dim,:] = temp_res  
        print("time : ",time.time() -start )
        print("shape of statistical res is: ",res.shape)
        self.stat_pmf = res 
        self.entropy_bottleneck.stat_pmf = res   



    def forward(self, x, training = False):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_hat, z_likelihoods , z_probability= self.entropy_bottleneck(z, training)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, training = training)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "probability": z_probability
        }

    def load_state_dict(self, state_dict):
        """
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        """
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=True, device = torch.device("cpu")):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update()
        return updated


    def compress_during_training(self, x, device = torch.device("cuda")):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress_during_training(z, device = device)
        z_hat = self.entropy_bottleneck.decompress_during_training(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_strings = self.gaussian_conditional.compress_during_training(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress_during_training(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress_during_training(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress_during_training(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



    def compress(self, x, means = None, device = torch.device("cpu")):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z, means = means)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


###########################################################################################################  
###########################################################################################################   
############################################# MEAN SCALE HYPERPRIOR #######################################
###########################################################################################################
###########################################################################################################




class ICMEMeanScaleHyperprior(ICMEScaleHyperprior):
    """
    """

    def __init__(self, N, M,extrema = 30, **kwargs):
        super().__init__(N, M, extrema = 30, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x, training):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods, probability = self.entropy_bottleneck(z, training)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat, training = training)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "probability": probability
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress_during_training(z)
        z_hat = self.entropy_bottleneck.decompress_during_training(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress_during_training(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress_during_training(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress_during_training(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}






###########################################################################################################  
###########################################################################################################   
############################################# JOINT AUTOREGRESSIVE ########################################
###########################################################################################################
###########################################################################################################


class ICMEJointAutoregressiveHierarchicalPriors(ICMEScaleHyperprior):


    def __init__(self, N=192, M=192,extrema = 30, **kwargs):
        super().__init__(N=N, M=M,extrema = extrema , **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, training = False):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods, probability = self.entropy_bottleneck(z, training) 
        params = self.h_s(z_hat)
        # check if self.training combacia 
        y_hat = self.gaussian_conditional.quantize(y,training)
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat,  training = training)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "probability": probability
        }

    def define_statistical_pmf(self, dataloader, device =torch.device("cuda"),  idx = 100):
        res = torch.zeros((self.N,2*self.extrema +1)).to(device)     
        start = time.time() 
        cc = 0         
        with torch.no_grad():
            for i,d in enumerate(dataloader):
                if i > idx - 1:
                    break
                cc += 1
                d = d.to(device)
                y = self.g_a(d) 
                out_enc= self.h_a(y)
                bs, ch, w,h = out_enc.shape
                
                out_enc = out_enc.round().int() #these dshould be the latent space
                out_enc = out_enc.reshape(ch,bs,w*h)
                prob = self.entropy_bottleneck._probability(out_enc)
                res  += prob 
        res = res/cc
        self.entropy_bottleneck.stat_pmf = res 
        self.entropy_bottleneck.pmf = res
        return res



    @classmethod
    def from_state_dict(cls, state_dict):

        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress_during_training(self, x,  device = torch.device("cuda")):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn("Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).")

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress_during_training(z, device = device) # cambiare
        z_hat = self.entropy_bottleneck.decompress_during_training(z_strings, z.size()[-2:]) # cambiare

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)  

                y_crop = y_crop[:, :, padding, padding]
                
                y_q = self.gaussian_conditional.quantize(y_crop,False, means = means_hat)
                #y_q = self.gaussian_conditional.quantize(y_crop,"symbols", means_hat) # cambiare 
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets) # fare check

        string = encoder.flush()
        return string

    def decompress_during_training(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn("Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).")



        z_hat = self.entropy_bottleneck.decompress_during_training(strings[1], shape) # cambiare
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros((z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),device=z_hat.device,)

        for i, y_string in enumerate(strings[0]):  # leggre che succede
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding): # leggere che succede
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(y_crop,self.context_prediction.weight,bias=self.context_prediction.bias,)
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                
                rv = self.gaussian_conditional.dequantize(rv, means = means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


###########################################################################################################  
###########################################################################################################   
################################################# CHENG 1 #################################################
###########################################################################################################
###########################################################################################################



class ICMECheng2020Anchor(ICMEJointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.
    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.
    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net






###########################################################################################################  
###########################################################################################################   
################################################# CHENG ATTENTION #########################################
###########################################################################################################
###########################################################################################################



class ICMECheng2020Attention(ICMECheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.
    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.
    Args:
        N (int): Number of channels
    """

    def __init__(self,N=192,  **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )