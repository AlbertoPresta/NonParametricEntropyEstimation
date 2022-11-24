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


    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=config["cfg"]["trainer"]["lr"],
    )

    return optimizer



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
        #bpp_y = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        bpp_y = bpp #len(out_enc["strings"][0][0]) * 8.0 / num_pixels
        bpp_z = bpp # len(out_enc["strings"][1][0]) * 8.0 / num_pixels
        #bpp = bpp_y + bpp_z
        return bpp, bpp_y, bpp_z
 


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compress_with_ac(model, test_dataloader, device,epoch):
    model.update()
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()
    timing_all = AverageMeter()
    timing_enc = AverageMeter()
    timing_dec = AverageMeter() 
    bpp_gauss = AverageMeter()
    bpp_hype = AverageMeter()   

    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
            d = d.to(device)
            start_all = time.time() 
            out_enc = model.compress_during_training(d)# bit_stream is the compressed, output_cdf needs for decoding 
            
            enc_comp = time.time() - start_all
            timing_enc.update(enc_comp)
            start = time.time()   
                       
            out_dec = model.decompress_during_training(out_enc["strings"], out_enc["shape"])
            
            timing_dec.update(time.time() - start)
            timing_all.update(time.time() - start_all)
            
            #selfout_dec["x_hat"].clamp_(0.,1.)
            bpp, bpp_gaussian , bpp_hyperprior = bpp_calculation(out_dec, out_enc)
            bpp_gauss.update(bpp_gaussian)
            bpp_hype.update(bpp_hyperprior)
            bpp_loss.update(bpp)
            psnr.update(compute_psnr(d, out_dec["x_hat"]))
            mssim.update(compute_msssim(d, out_dec["x_hat"]))   
    
    log_dict = {
            "test":epoch,
            "test/bpp_with_ac": bpp_loss.avg,
            "test/bpp_with_ac_gaussian":bpp_gauss.avg,
            "test/bpp_with_ac_hype":bpp_hype.avg,
            "test/psnr_with_ac": psnr.avg,
            "test/mssim_with_ac":mssim.avg,
            
            "test/timing_all":timing_all.avg,
            "test/timing_enc":timing_enc.avg,
            "test/timing_dec":timing_dec.avg   
    }
    
    wandb.log(log_dict)
    return bpp_loss.avg
                
    
    


def plot_likelihood_baseline(net, device, epoch,n = 1000, dim = 0):
    quantiles = torch.tensor([-20,0,20]).to(device)
    quantiles = quantiles.repeat(net.M).reshape([net.M,1,3])
    minimo = torch.min(quantiles[:,:,0]).item()
    massimo = torch.max(quantiles[:,:,2]).item()
    space = (massimo - minimo)/n
    x_values = torch.arange(minimo, massimo, space)
    sample = x_values.repeat(net.M, 1).unsqueeze(1).to(device) # [192,1,1000]
    
    y_values = net.entropy_bottleneck._likelihood(sample, False)[dim, :].squeeze(0) #[1000]
    data = [[x, y] for (x, y) in zip(x_values,y_values)]
    table = wandb.Table(data=data, columns = ["x", "p_y"])
    wandb.log({"likelihood at dimension " + str(dim): wandb.plot.line(table, "x", "p_y", title='likelihood function at dimension ' + str(dim))})
    
    
"""
def compute_and_plot_bitrate(probability, device, num_pixels, like_dim, epoch):
    # probability has dimension [192,NL] 
    res = -torch.sum(probability*(torch.log(probability)/math.log(2)))*like_dim/(num_pixels)
    log_dict = {
    "test":epoch,
    "test/entropy statistcal": res.item()
    }
    wandb.log(log_dict) 
"""  



def plot_latent_space_frequency(model, test_dataloader, device,dim = 0, test = True):
        model.eval()
        extrema = model.entropy_bottleneck.extrema
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

        if test is True:
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,model.entropy_bottleneck.pmf[dim,:])]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"model probability dix at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='model probability at dimension ' + str(dim))}) 
            
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,res)]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"test frequency statistics at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='test frequency statistics at dimension ' + str(dim))})         
        else:
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,res)]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"train frequency statistics at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='train frequency statistics at dimension ' + str(dim))})   
        return res
             



def plot_hyperprior_latent_space_frequency(model, test_dataloader, device,dim = 0, test = True):
        model.eval()
        extrema = model.entropy_bottleneck.extrema
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
                out_enc = model.h_a(torch.abs(out_enc))
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

        if test is True:
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,model.entropy_bottleneck.pmf[dim,:])]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"model probability dix at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='model probability at dimension ' + str(dim))}) 
            
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,res)]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"test frequency statistics at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='test frequency statistics at dimension ' + str(dim))})         
        else:
            x_values = torch.arange(-extrema, extrema + 1)      
            data = [[x, y] for (x, y) in zip(x_values,res)]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({"train frequency statistics at dimension" + str(dim): wandb.plot.scatter(table, "x", "p_y", title='train frequency statistics at dimension ' + str(dim))})   
        return res
                          
                
def plot_different_values(model, test_dataloader, device, M = 192 ,dim = 191):
    cont = torch.zeros(M)
    
    unique_val_ch = []
    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            out_enc = model.g_a(d)
            bs, ch, w,h = out_enc.shape
            print("canali:   ",ch)
            out_enc = out_enc.reshape(ch,bs,w*h)
            y = model.entropy_bottleneck.sos(out_enc,-1)
            for j in range(ch):
                print(j)
                unique_val, number_val = torch.unique(y[j,:,:],return_counts=True)                    
                cont[j] += unique_val.shape[0] 
                if j == dim:
                    unique_val_ch += unique_val
    
    cont = cont/len(test_dataloader)
    cont = cont.tolist() # convert to list 
    

    x_values = torch.arange(model.M).tolist()
    data = [[x, y] for (x, y) in zip(x_values, cont)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"number of values per channel" : wandb.plot.scatter(table, "x", "y",
                                 title="number of values per channel")})
           
           
           
from scipy.spatial.distance import jensenshannon

def compute_prob_distance(pmf1, pmf2, epoch,dim):
    r = jensenshannon(pmf1.detach().cpu(),pmf2.detach().cpu())   
    log_dict = {
        "test":epoch,
        "test/pmf_distance_" + str(dim): r  }   
    wandb.log(log_dict)
