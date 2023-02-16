import shutil 
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.utils import make_grid
import time
import math 
from pytorch_msssim import ms_ssim
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, config):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=config["cfg"]["trainer"]["lr"],
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=config["cfg"]["trainer"]["aux_lr"],
    )
    return optimizer, aux_optimizer






def save_checkpoint(state, is_best, filename, filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net, out_enc = None):
    if out_enc is None: 
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in out_net['likelihoods'].values()).item()
    else:
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp
    


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp

def compress_with_ac(model, test_dataloader, device,epoch):
    model.update(force = True)
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()
    timing_all = AverageMeter()
    timing_enc = AverageMeter()
    timing_dec = AverageMeter()
    start = time.time()    
    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
            d = d.to(device) 
            start_all = time.time()  
            out_enc = model.compress(d)# bit_stream is the compressed, output_cdf needs for decoding 
            enc_comp = time.time() - start_all
            timing_enc.update(enc_comp)
            start = time.time()  
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            timing_dec.update(time.time() - start)
            timing_all.update(time.time() - start_all)
            #out_dec["x_hat"].clamp_(0.,1.)
            bpp = bpp_calculation(out_dec, out_enc)
            bpp_loss.update(bpp)
            psnr.update(compute_psnr(d, out_dec["x_hat"]))
            mssim.update(compute_msssim(d, out_dec["x_hat"]))   
    
    log_dict = {
            "test":epoch,
            "test/bpp_with_ac": bpp_loss.avg,
            "test/psnr_with_ac": psnr.avg,
            "test/mssim_with_ac":mssim.avg 
    }
    
    wandb.log(log_dict)
                


def plot_likelihood_baseline(net, device, epoch,n = 1000, dim = 0):
    #minimo = torch.min(net.entropy_bottleneck.quantiles[:,:,0]).item()
    #massimo = torch.max(net.entropy_bottleneck.quantiles[:,:,2]).item()
    minimo = -30
    massimo = 30
    space = (massimo - minimo)/n
    x_values = torch.arange(minimo, massimo)
    sample = x_values.repeat(net.N, 1).unsqueeze(1).to(device) # [192,1,1000]
    
    y_values = net.entropy_bottleneck._likelihood(sample)[dim, :].squeeze(0) #[1000]
    data = [[x, y] for (x, y) in zip(x_values,y_values)]
    table = wandb.Table(data=data, columns = ["x", "p_y"])
    wandb.log({"llikelihood at dimesion " + str(dim): wandb.plot.scatter(table, "x", "p_y", title="likelihood at dimesion " + str(dim))})
    
    


def plot_latent_space_frequency(model, test_dataloader, device,epoch,dim = 0, test = True):
        model.eval()
        extrema = 30
        if test is True:
            res = torch.zeros((len(test_dataloader),2*extrema +1)).to(device)
        else:
            res = torch.zeros((1000,2*extrema +1)).to(device)
        cont = 0
        with torch.no_grad():
            for i,d in enumerate(test_dataloader):
                if i > 999 and test is False:
                    break
                cont = cont + 1
                d = d.to(device)
                out_enc = model.g_a(d) 
                bs, ch, w,h = out_enc.shape
                out_enc = out_enc.round().int() #these dshould be the latent space
                out_enc = out_enc.reshape(bs,ch,w*h)
                unique, val = torch.unique(out_enc[0,dim,:], return_counts = True)
                dict_val =  dict(zip(unique.tolist(), val.tolist()))
                cc = 0 
                for j in range(-extrema, extrema + 1):
                    if j not in unique:
                        res[i,cc]  = 0 
                        cc = cc + 1
                    else:
                        res[i,cc] = dict_val[j]
                        cc = cc +1

        
        
        res = torch.sum(res, dim = 0)
        res = res.reshape(-1)
        res = res/torch.sum(res)

        x_values = torch.arange(-extrema, extrema + 1)      
        data = [[x, y] for (x, y) in zip(x_values,res)]
        table = wandb.Table(data=data, columns = ["x", "p_y"+ str(dim) + " " + str(epoch)])
        wandb.log({"counter/test frequency statistics at dimension " + str(dim) + " " + str(epoch): wandb.plot.scatter(table, "x", "p_y"+ str(dim) + " " + str(epoch), title='test frequency statistics at dimension ' + str(dim) + " " + str(epoch))}) 