import torch 
import wandb 
import numpy as np  
import os 
import argparse
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from Datasets.dataset import Datasets, TestKodakDataset, VimeoDatasets
from torch.utils.data import DataLoader
import os
from os.path import join, exists, isfile
from compressai.zoo import *
import wandb 
import time
from compAi.training.icme.loss import EntropyDistorsionLoss
from compAi.training.icme.step  import train_one_epoch, test_epoch
from compAi.training.icme.utility import plot_likelihood, CustomDataParallel, configure_optimizers, save_checkpoint,  plot_likelihood_baseline, plot_latent_space_frequency, plot_hyperprior_latent_space_frequency,compute_prob_distance, compress_with_ac
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior, ICMEMeanScaleHyperprior, ICMEJointAutoregressiveHierarchicalPriors
from compAi.utils.parser import parse_args, ConfigParser
import collections


model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "icme2023-factorized": FactorizedICME,
    "icme2023-hyperprior": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior

}

def extract_pmf_from_baseline(net):   
    ep = net.entropy_bottleneck
    medians = ep.quantiles[:, 0, 1]
    minima = medians - ep.quantiles[:, 0, 0]
    minima = torch.ceil(minima).int()
    minima = torch.clamp(minima, min=0)
    maxima = ep.quantiles[:, 0, 2] - medians
    maxima = torch.ceil(maxima).int()
    maxima = torch.clamp(maxima, min=0)
    pmf_start = medians - minima
    pmf_length = maxima + minima + 1
    max_length = pmf_length.max().item()
    device = pmf_start.device
    samples_c = torch.arange(-30, 31, device=device)   
    samples = samples_c.repeat(192,1).unsqueeze(1)
    half = float(0.5)
    lower = ep._logits_cumulative(samples - half, stop_gradient=True)
    upper = ep._logits_cumulative(samples + half, stop_gradient=True)
    sign = -torch.sign(lower + upper)
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]     
    samples = samples[:,0,:]
    return pmf

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def load_pretrained_baseline(architecture, path,device = torch.device("cpu")):  
    print("architecture!!",path)
    if (architecture not in model_architectures):
        raise RuntimeError("Pre-trained model not yet available")   
    elif isfile(path) is False:
        raise RuntimeError("path is wrong!")
    mod_load = torch.load(path,map_location= device ) # extract state dictionary
    net= from_state_dict(architecture, mod_load["state_dict"])
    if "icme" in path.split("/")[-1]:
        net.entropy_bottleneck.pmf = mod_load["pmf"]
        net.entropy_bottleneck.stat_pmf = mod_load["stat_pmf"]
    return net



def from_state_dict(arch, state_dict):
    """Return a new model instance from `state_dict`."""
    N = state_dict["g_a.0.weight"].size(0)
    M = state_dict["g_a.6.weight"].size(0)
    if "icme" in arch:
        net = model_architectures[arch](N, M)
    else: 
        net = model_architectures[arch](quality = 1)
    net.load_state_dict(state_dict)
    return net

def load_model(path_models, name_model, type_mode, device = torch.device("cpu")):

    if "icme" in name_model:
        complete_path = join(path_models,name_model + ".pth.tar")
        net = load_pretrained_baseline(type_mode, complete_path, device  = device)
        net.update(device = device)
    else: # use only baseline
        
        complete_path = join(path_models,name_model + ".pth.tar")
        net = load_pretrained_baseline(type_mode, complete_path, device  = device)    
        net.update()  
    return net


def compute_pmf(net, dataloader, device = torch.device("cpu")): 
    res = torch.zeros(net.M, net.entropy_bottleneck.levels.shape[0]).to(device)
    # calculate the pmf using training dataloader
    cc = 0
    with torch.no_grad():

        for i,d in enumerate(dataloader):
            if i<500:   
                if i%10:
                    print(i)                
                cc += 1
                d = d.to(device)
                x = net.g_a(d)
                bs,ch,w,h = x.shape  
                x = x.reshape(ch,bs,w*h)                       
                outputs = net.entropy_bottleneck.quantize(x,  False)
                prob = net.entropy_bottleneck._probability(outputs)

                res += prob
            else:
                break
        res = res/cc
        return  res
    


def compute_and_plot_latentspace_channels(net,type, dataloader = None, test = False, device = torch.device("cpu")):
    if "icme" in type:
        if dataloader is None:
            # plot pmf values  
            x_values = net.entropy_bottleneck.levels
            for i in range(2):
                pmf = net.entropy_bottleneck.pmf[i,:]                 
                data = [[x, y] for (x, y) in zip(x_values,pmf)]
                table = wandb.Table(data=data, columns = ["x", "p_y"])
                wandb.log({'likelihood function at dimension ' + str(i): wandb.plot.scatter(table, "x", "p_y" , title='likelihood function at dimension ' + str(i))})
        else:       # here we take the averageon the test set
            with torch.no_grad():
                res = torch.zeros(net.M, net.entropy_bottleneck.levels.shape[0]).to(device)
                cc = 0
                for j,d in enumerate(dataloader): 
                    if j < 30000:
                        if j%700==0:
                            print(j)               
                        cc += 1
                        d = d.to(device)
                        x = net.g_a(d)
                        bs,ch,w,h = x.shape  
                        x = x.reshape(ch,bs,w*h)                       
                        outputs = net.entropy_bottleneck.quantize(x,  False)
                        prob = net.entropy_bottleneck._probability(outputs)

                        res += prob  
                    else: break
                    
                res = res/cc  #questa è la distribuzione  del test set
                x_values = net.entropy_bottleneck.levels
                for j in range(2):
                    print("----> ",j)
                    pmf = res[j,:]
                    data = [[x, y] for (x, y) in zip(x_values,pmf)]
                    table = wandb.Table(data=data, columns = ["x", "p_y"])
                    if test is True:
                        wandb.log({'test distribution at dimension ' + str(j): wandb.plot.scatter(table, "x", "p_y" , title='test distribution at dimension' + str(j))})
                    else:
                        wandb.log({'train distribution at dimension ' + str(j): wandb.plot.scatter(table, "x", "p_y" , title='train distribution at dimension' + str(j))})
    else:
        # plot pmf values  
        x_values = net.entropy_bottleneck.levels
        for i in range(2):
            pmf = net.entropy_bottleneck.pmf[i,:]                 
            data = [[x, y] for (x, y) in zip(x_values,pmf)]
            table = wandb.Table(data=data, columns = ["x", "p_y"])
            wandb.log({'likelihood function at dimension ' + str(i): wandb.plot.scatter(table, "x", "p_y" , title='likelihood function at dimension ' + str(i))})        
        
                    
                        
def main(config):
    
    path_models = config["path_models"]#"/scratch/pretrained_models"
    name_model = "icme2023-factorized_0018"
    type_mode = "icme2023-factorized"
   
    net = load_model(path_models, name_model, type_mode)
    
    
    td_path  = config["dataset"]["train_dataset"]
    file_txt = config["dataset"]["file_txt"]
    img_size = config["dataset"]["image_size"] 
    test_td_path = config["dataset"]["test_dataset"] #config["dataset"]["test_dataset"]
    train_dataset = VimeoDatasets(td_path, file_txt,image_size = img_size) 
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1,shuffle=True,pin_memory=True,num_workers=4)

    
    
    test_dataset = TestKodakDataset(data_dir=test_td_path)
    test_dataloader = DataLoader( test_dataset,batch_size=1,num_workers=4,shuffle=False,pin_memory=True)

    #res = compute_pmf(net, train_dataloader)
    compute_and_plot_latentspace_channels(net,type_mode)
    print("---")
    compute_and_plot_latentspace_channels(net,type_mode,dataloader = test_dataloader, test = True)
    print("---")
    compute_and_plot_latentspace_channels(net,type_mode,dataloader = train_dataloader, test = False)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="configuration/test.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')



    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--ql', '--quality'], type=int, target='arch;quality'),
         CustomArgs(['--lmb', '--lambda'], type=float, target='cfg;trainer;lambda'),
        CustomArgs(['--pw', '--power'], type=float, target='cfg;trainer;power')


    ]
    
    wandb.init(project="inference", entity="albertopresta")
    config = ConfigParser.from_args(args, wandb.run.name, options)
    wandb.config.update(config._config)
    main(config)
    