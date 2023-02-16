import yuvio 
import numpy as np 
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
import yuvio
import torch
import shutil
from torchvision.utils import save_image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import cv2
from scipy.spatial.distance import jensenshannon
from PIL import Image
from torch.utils.data import DataLoader
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior,ICMECheng2020Attention, ICMEMeanScaleHyperprior, ICMEJointAutoregressiveHierarchicalPriors
from pytorch_msssim import ms_ssim
from Datasets.dataset import Datasets, TestKodakDataset, VimeoDatasets
import math
from compressai.zoo import *
from matplotlib.lines import Line2D

model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "hyper": bmshj2018_hyperprior,
    "minnen2019":mbt2018,
    "cheng": cheng2020_attn,
    "icme2023-factorized": FactorizedICME,
    'icme2023-cheng':ICMECheng2020Attention,
    "icme2023-hyper": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior,
    "icme2023-joint":ICMEJointAutoregressiveHierarchicalPriors

}


def extract_pmf_model(model):
    # Check if we need to update the bottleneck parameters, the offsets are
    # only computed and stored when the conditonal model is update()'d.
 
    medians = model.entropy_bottleneck.quantiles[:, 0, 1]

    minima = medians - model.entropy_bottleneck.quantiles[:, 0, 0]
    minima = torch.ceil(minima).int()
    minima = torch.clamp(minima, min=0)

    maxima = model.entropy_bottleneck.quantiles[:, 0, 2] - medians
    maxima = torch.ceil(maxima).int()
    maxima = torch.clamp(maxima, min=0)

    model.entropy_bottleneck._offset = -minima

    pmf_start = medians - minima
    pmf_length = maxima + minima + 1

    max_length = pmf_length.max().item()
    device = pmf_start.device
    samples = torch.arange(max_length, device=device)

    samples = samples[None, :] + pmf_start[:, None, None]

    half = float(0.5)

    lower = model.entropy_bottleneck._logits_cumulative(samples - half, stop_gradient=True)
    upper = model.entropy_bottleneck._logits_cumulative(samples + half, stop_gradient=True)
    sign = -torch.sign(lower + upper)
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]
    return pmf


def compute_prob(net,dataloader,device, idx):
    
    res = torch.zeros((net.M,2*net.extrema +1)).to(device)  
    cc = 0
    
    with torch.no_grad() :
        for i,d in enumerate(dataloader):

            if i > idx: 
                break 
            else:
                cc += 1
                d = d.to(device)
                y = net.g_a(d)
                bs, ch,w,h = y.shape
                y = y.reshape(ch,bs, w*h)
                prob = net.entropy_bottleneck._probability(y)
                res += prob 

    return res/cc     
            
            
            
def load_pretrained_net( mod_load, path_models, architecture, type_mode, ml):
    if "icme" in type_mode:
        N = mod_load["N"]
        M = mod_load["M"] 
        #model = architecture()
        model = architecture(N = N, M = M )
        model = model.to(ml)
        model.load_state_dict(mod_load["state_dict"])  
        model.entropy_bottleneck.pmf = mod_load["pmf"]
        model.update( device = torch.device("cpu"))        
        return model
    else:
        model = load_pretrained_baseline(type_mode, path_models, device  = ml)
        model.update()
        
        return model
        


def main():
    td_path  = "/Users/albertopresta/Desktop/icme/vimeo_triplet/sequences"
    file_txt = "/Users/albertopresta/Desktop/icme/vimeo_triplet/tri_trainlist.txt"
    train_dataset = VimeoDatasets(td_path, file_txt,image_size = 256) 
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1,shuffle=True,pin_memory=True,num_workers=4)
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    path_models = "/Users/albertopresta/Desktop/icme/models/factorized/icme/icme2023-factorized_0018.pth.tar"    
        
    test_dataset = TestKodakDataset(data_dir=path_images)
    test_dataloader = DataLoader( test_dataset,batch_size=1,num_workers=4,shuffle=False,pin_memory=True)

    checkpoint = torch.load(path_models, map_location= torch.device("cpu"))
    check = torch.load("/Users/albertopresta/Desktop/icme/models/factorized/icme/bmshj2018-factorized_0018.pth.tar", map_location= torch.device("cpu"))
    model = FactorizedICME(N = 128, M = 192 )
    model_base = bmshj2018_factorized(quality = 1)
    model_base.load_state_dict(check["state_dict"])
    model_base.update()
    pmf_base = extract_pmf_model(model_base)
    
    
    
    model = model.to(torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])  
    model.entropy_bottleneck.pmf = checkpoint["pmf"]
    model.update( device = torch.device("cpu"))   
    print("inizio")
    test_res = compute_prob(model,test_dataloader,torch.device("cpu"), 100)
    print("fine", test_res.shape)

    #print("inizio")
    #test_res_base = compute_prob(model_base,test_dataloader,torch.device("cpu"), 100, pmf_base.shape[-1])
    #print("fine", test_res.shape)   
    
    for i in [10000]:
        print("-------------------------  ",i," ---------------------------")
        tmp = compute_prob(model,train_dataloader,torch.device("cpu"), i)
        
        rr = 0.0
        for j in range(192):
            r = jensenshannon(test_res[j,:],tmp[j,:])
            if r < 0.0002:
                r = 0
            rr += r 
        print("rr for ",i,": ",rr)
    
if __name__ == "__main__":
    main()
    
