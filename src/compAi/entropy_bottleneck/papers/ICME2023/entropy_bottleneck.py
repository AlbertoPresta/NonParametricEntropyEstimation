import warnings

from typing import Any, Callable, List, Optional, Tuple, Union
#import torchac 
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound
import warnings
warnings.filterwarnings("ignore")

class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any):
    raise NotImplementedError()


class EntropyModel(nn.Module):
    r"""Entropy model base class.
    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        delta,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()
        self.delta = delta
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(self, inputs, training,  means = None, comp = False):

        if training:
            half = float((self.delta/2))
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise           
            return inputs
        else:
            outputs = inputs.clone()           
            if means is not None:
                outputs -= means
            
            
            if self.delta == 1:
                outputs = torch.round(outputs)  
            else: 
                outputs = self.customize_quantize(outputs)
            
            if means is not None and comp is False:
                outputs += means
                             
            return outputs


    
    def customize_quantize(self,x):
        b = self.levels 
        #torch.sum(torch.sign(torch.relu(1 - (2/self.delta)*torch.abs(x - b[:,None])))*b[:,None], dim = 0)
        return torch.sum(torch.sign(torch.relu(1 - (2/self.delta)*torch.abs(x - b[None,:,None].to(x.device))))*b[None,:,None].to(x.device), dim = 1).unsqueeze(1)

    def transform_map(self,x,map_float_to_int):
        if x in map_float_to_int.keys():
            return map_float_to_int[x]
        else:
            # find the closest key and use this
            keys = np.asarray(list(map_float_to_int.keys()))
            i = (np.abs(keys - x)).argmin()
            key = keys[i]
            return map_float_to_int[key]

    
    def dequantize(self,inputs,dtype = torch.float, means = None):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs



    def _pmf_to_cdf(self, pmf, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        tail_mass = torch.zeros(pmf.shape[0],1).to(pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p, tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf

        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")
    
    def compress_during_training(self, inputs, indexes, means = None):
        bs,ch,w,h = inputs.shape 
        inputs = inputs.reshape(ch,1,bs*w*h)
         
        symbols = self.quantize(inputs, False, means = means)
        symbols = symbols.reshape(bs,ch,w,h)
        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings
    


    
    def decompress_during_training(self,strings,indexes,dtype = torch.float, means = None):
        cdf = self._quantized_cdf.to(indexes.device)
        self._cdf_length.to(indexes.device)
        self._offset.to(indexes.device)
        outputs = cdf.new_empty(indexes.size())
        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means = means)
        return outputs
    

class  EntropyBottleneck(EntropyModel):
    """
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        extrema: int = 30,
        init_scale: float = 10,
        power = 1,
        delta = 1,
        **kwargs: Any,
    ):
        super().__init__(delta,*args, **kwargs)
        self.delta = delta
        self.channels = int(channels)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.stat_pmf = None
        self.power = power
        self.pmf = None
        self.epsilon = 1e-7
        self.extrema = extrema
        self.levels = torch.arange(-self.extrema, self.extrema + self.delta, self.delta)
        self.build_maps()

    def _get_medians(self):
        print()
        medians = self.quantiles[:, :, 1:2]
        return medians
      
    def pmf_to_cdf(self, prob_tens = None):
        if prob_tens is None:
            cdf = self.pmf.cumsum(dim=-1)
            spatial_dimensions = self.pmf.shape[:-1] + (1,)
            zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
            cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
            cdf_with_0 = cdf_with_0.clamp(max=1.)
            return cdf_with_0
        else:
            cdf = prob_tens.cumsum(dim= -1)
            cdf_with_0 = torch.zeros(self.M, cdf.shape[1] + 1)
            for i in range(self.M):
                cdf_with_0[i,1:] =  cdf[i]
            return cdf_with_0    
    
    def update(self, device = torch.device("cuda"),stat_pmf = None):

        if stat_pmf is not None:
            self.pmf = stat_pmf
        self.cdf = self.pmf_to_cdf()       
        tail_mass = torch.zeros(self.channels).to(device)
        pmf_length = torch.zeros(self.channels).to(device)
        pmf_length += self.levels.shape[0]  # ogni pmf ha lo stesso numero di lunghezza
        pmf_start = torch.zeros(self.channels).to(device)
        pmf_start += -self.extrema  # partono tutte dal minimo       
        pmf_length = pmf_length.int()
        pmf_start = pmf_start.int() 
        max_length = pmf_length.max().int()
        quantized_cdf = self._pmf_to_cdf(self.pmf,pmf_length, max_length)
        self._quantized_cdf.data= quantized_cdf
        self.quantiles = torch.nn.Parameter(torch.zeros(self.channels,1,3).to(device))       
        self._cdf_length = pmf_length + 2
        self._offset = torch.zeros(self.channels).to(device) - self.extrema      
        return True
        
        
    def retrieve_cdf(self, prob_tens = None):
        cdf = self.pmf.cumsum(dim=-1)
        spatial_dimensions = self.pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)
        return cdf_with_0



    def _likelihood(self, x):

        d = torch.abs(x - self.levels[:,None].to(x.device))
        d = (2/self.delta)*d
        d = torch.pow(d + self.epsilon, self.power)
        dist = torch.relu(1 - d)            
        #dist = torch.relu(1 - (2**self.power)*torch.abs((x - self.levels[:,None].to(x.device))**self.power))
        pmf = self.pmf[:,:,None].to(x.device)
        likelihood = (pmf + self.epsilon)*dist #[192,NL,8291]
        likelihood = torch.sum(likelihood,dim = 1).unsqueeze(1)
        return likelihood #[192,1,___]





    def _probability(self, x: Tensor): 
        """
        function that cquantomputes the distribution of each channel  
        NL = number of levels 
        NC = number of channels       
        shape of x: [192,1,___]
        """           

        d = torch.abs(x - self.levels[:,None].to(x.device))
        d = (2/self.delta)*d
        d = torch.pow(d + self.epsilon, self.power)
        d = torch.relu(1 - d)                
        #d = torch.relu(1 - (2**self.power)*torch.pow(torch.abs((x - self.levels[:,None].to(x.device))),self.power))  #[192,NL,8192]  
        #d = torch.relu(torch.exp(-(x - self.sos.cum_w[:,None].to(x.device))**2) - 0.01)
        y_sum = torch.sum(d,dim = 2) #[NC,NL]
        y = torch.sum(y_sum,dim = 1) #[NC,1]
        return y_sum/y[:,None]  #[numero canali, numero di livelli]

    
    
    def _hemp(self, x: Tensor, training = True):          
        d = torch.exp(-(x - self.levels[:,None].to(x.device))**2) #[192,NL,8192] 
        d_sum = torch.sum(d, dim = 1)   # sum over the channels ---> [192,8192]   
        y = d/d_sum[:,None,:]   #192,NL,8192 . Questo rappresenta esattamente la formula di hemp per ogni punto
        return y
      

    def forward(self, x: Tensor, training):
        #self.levels = torch.arange(-self.extrema, self.extrema + self.delta.item(), self.delta.item())
        # x from B x C x ... to C x B x ...
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)
        # Add noise or quantize

        outputs = self.quantize(values,training)
        if not torch.jit.is_scripting():
            probability = self._probability(outputs) 
            probability = self.likelihood_lower_bound(probability)
            if training:
                self.pmf = probability # update pmf only when training
                #likelihood = self.compute_rate(outputs)
                likelihood = self._likelihood(outputs)
                #pr = self.likelihood_lower_bound(pr)

                self.stat_pmf = probability
                #v = self.quantize(values,training)
                #likelihood = self._likelihood(outputs)
                #likelihood = pr
            else:
                
                likelihood = self._likelihood(outputs)   
                #likelihood = self.compute_rate(outputs)         
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
                
        else:
            raise NotImplementedError()
        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood, probability 

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)
    
    def compress_during_training(self, x, device):
        self.update(device = device)
        indexes = self._build_indexes(x.size())
        return super().compress_during_training(x, indexes)

    
    
    def compute_rate(self,inputs):

        a,_,b = inputs.shape
        inputs = inputs.reshape(-1).to(inputs.device)
        inputs = inputs.unsqueeze(1).to(inputs.device)
        ap = self.levels.to(inputs.device)
        apl = torch.zeros(ap.shape[0] + 1).to(ap.device) - ap[0] - self.delta # [l + 1]
        apl[1:] = ap 
        apr = torch.zeros(ap.shape[0] + 1).to(ap.device) + ap[-1] + self.delta # [l + 1]
        apr[:-1] = ap  
        
        li = inputs < apr  
        ri = inputs >= apl
        
        one_hot_level = torch.logical_and(li,ri).to(inputs.device)    # [190000,level + 1]
        
        one_hot_level = one_hot_level.reshape(a,b,-1) #[192,100,l +1]
        one_hot_level = torch.permute(one_hot_level, (0,2,1)) # [192,l+1,100]
        
        pmf_l = torch.zeros(self.pmf.shape[0],self.pmf.shape[1] + 1).to(inputs.device)
        pmf_l[:,1:] = self.pmf
        #pmf_l = pmf_l.repeat(192).reshape(192,-1).to(inputs.device) #[192,l+1]

        pmf_r = torch.zeros(self.pmf.shape[0],self.pmf.shape[1] + 1).to(inputs.device)
        pmf_r[:,:-1] = self.pmf
        #pmf_r = pmf_r.repeat(192).reshape(192,-1).to(inputs.device) #[192,l+1]  
        
        inputs = inputs.reshape(a,1,b) #[192,1,100]
        p_x = (inputs - apl[:,None])/(apr[:,None] - apl[:,None])
        p_x = p_x*(pmf_r[:,:,None] - pmf_l[:,:,None]) + pmf_l[:,:,None] #[192,L+1, 100]
        
        p_x = torch.relu(p_x)
        p_x = p_x*one_hot_level
        p_x = torch.sum(p_x,dim = 1).unsqueeze(1) #[192,1,100]
        return p_x
        
        
        
              
        
        
        
        
        
    
    
    def compress(self, x,means = None):

        x = self.quantize(x, False, means = means) 
        print("primo controllo: ",torch.unique(x, return_counts = True))
        x = x.detach().apply_(lambda x: self.transform_map(x, self.float_to_int))  
        print("secondo controllo: ",torch.unique(x, return_counts = True))
        symbols = x #[1,192,32,48]
        M = symbols.size(1)        
        symbols = symbols.to(torch.int16)
        output_cdf = torch.zeros_like(symbols,dtype=torch.int16) 
        output_cdf = output_cdf[:,:,:,:,None] + torch.zeros(self.cdf.shape[1])
        for i in range(M):
            output_cdf[:,i,:,:,:] = self.cdf[i,:]      
        byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)
        if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
            raise ValueError("arithmetic codec did not work properly. Debug")   
        else:
            print("ok sono sopravvissuto")    
        return byte_stream, output_cdf
    
    
    def decompress_during_training(self, strings, size):       
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)  
        return super().decompress_during_training(strings, indexes)
    
    
    def decompress(self, byte_stream, output_cdf):
        outputs = torchac.decode_float_cdf(output_cdf, byte_stream)
        outputs = self.dequantize(outputs)
        #outputs= 
        #for i in range(self.M):            
        #    inputs[0,i,:,:] = inputs[0,i,:,:].apply_(lambda x: self.sos.map_cdf_sos[i][x])
        #outputs = inputs.type(torch.float)

        return outputs

    

    def build_maps(self):
        self.float_to_int = {}
        self.int_to_float = {}
        for i in range(self.levels.shape[0]):
            self.float_to_int[self.levels[i].item()] = i
            self.int_to_float[i] = self.levels[i].item()



######################################################################################
######################################################################################
######################### GAUSSIAN CONDITIONAL   #####################################
######################################################################################
######################################################################################



class GaussianConditional(EntropyModel):

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(1,*args, **kwargs)

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        if self._offset.numel() > 0 and not force:
            return False
        device = scale_table.device
        #self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.scale_table = torch.Tensor(tuple(float(s) for s in scale_table))
        self.scale_table = self.scale_table.to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()
        
        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs, scales, means = None):
        half = float(0.5)
        
        
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.lower_bound_scale(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def forward(self, inputs, scales, means = None,training = False):

        outputs = self.quantize(inputs, training, means = means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes