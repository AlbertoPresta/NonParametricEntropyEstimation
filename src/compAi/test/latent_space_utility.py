
import math
import io
import torch
import time
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import wandb
from collections import defaultdict
from pytorch_msssim import ms_ssim

import os
import pickle, gzip

import torchvision
import torch
import argparse
import json
from scipy.spatial.distance import jensenshannon
import argparse
import collections


from compressai.zoo import bmshj2018_factorized



from compAi.test.evaluate import *
from compAi.test.utility import create_net_dict



def define_samples(net, name):
    if "sos" in name:
        samples = net.entropy_bottleneck.sos.cum_w 
        if net.entropy_bottleneck.activation == "sigmoid":
            samples = samples.repeat(192,1).unsqueeze(1)
        else:
            samples = samples.unsqueeze(1)
        return samples 
    else:
        return -1
               





   
 
def prepare_dataset(img_list, ch = 3,image_size = 256):
    N = len(img_list)
    res = torch.zeros((N, ch , image_size, image_size))
    device = torch.device("cpu")
    for i in range(N):
        img = Image.open(img_list[i]).convert("RGB")
        x = transforms.Resize((image_size,image_size))(img)
        x = transforms.ToTensor()(x).to(device)
        res[i,:,:,:] = x
    return res
  
def create_dictionary(a,b,l):
    res = {}
    cot = 0
    for i in l:
        if i in a:
            res[i] = b[cot].item() 

            cot = cot + 1
        else:
            res[i] = 0
    return res 


def extract_permutation(x):
    perm = np.arange(len(x.shape))
    perm[0], perm[1] = perm[1], perm[0]
    inv_perm = np.arange(len(x.shape))[np.argsort(perm)] 
    return perm, inv_perm  





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
    return pmf, samples


   
    
def analyze_latentspace(net,img_list, sos = True, path = "../images/latent_space/baseline", samples = None, pmf = None):
    img_tensor = prepare_dataset(img_list)
    device = torch.device("cpu")
    with torch.no_grad():
        if sos:
            distance_av = []
            y = net.g_a(img_tensor) 
            perm , inv_perm = extract_permutation(y)
            pr = net.entropy_bottleneck.quantize(y, False, symbols = True)
            maps = net.entropy_bottleneck.sos.map_cdf_sos
            pr = pr.permute(*perm).contiguous()
            pr = pr.reshape(pr.size(0), 1, -1)
            pr = pr[:,0,:]            
            for i in range(net.M):
                a,b = torch.unique(pr[i],return_counts=True)
                pmf_dic = dict(zip(np.arange(0,pmf[i].shape[0]),list(pmf[i].numpy())))
                somma = torch.sum(b).item()
                b = (b/somma)#*100
                data_dic = create_dictionary(a,b,np.arange(0,pmf[i].shape[0]))
                if a.shape[0] > 1:
                    plot_distribution(data_dic,pmf_dic,i,path)
                distance_av.append(compute_prob_distance(data_dic,pmf_dic,0))

            plt.figure(figsize=(14, 6))
            plt.bar(np.arange(0,len(distance_av)),distance_av)  
            plt.grid()
            plt.xlabel('channels')
            plt.ylabel('JS_distance')
            pt = os.path.join(path,"dix.png") 
 
            plt.savefig(pt)

            print(np.mean(np.array(distance_av)))
        else:
            distance_av = []
            pmf, samples = extract_pmf_model(net)

            samples = samples[:,0,:].round().int()
            pmf = pmf
            yy = net.g_a(img_tensor)
            yy = net.entropy_bottleneck.quantize(yy, "symbols", means = net.entropy_bottleneck._get_medians())
            #medians = net.entropy_bottleneck._get_medians()
            yy = yy - net.entropy_bottleneck._get_medians()
            yy = yy.round()
            perm, inv_perm = extract_permutation(yy)
            yy = yy.permute(*perm).contiguous()
            yy = yy.reshape(yy.size(0), 1, -1)
            yy = yy[:,0,:]
            for i in range(yy.shape[0]):
                a,b = torch.unique(yy[i],return_counts=True)
                a = a.int()
                somma = torch.sum(b).item()
                b = (b/somma)
                data_dic = create_dictionary(a,b,samples[i].detach().numpy())
                    
                pmf_dic = dict(zip(samples[i].detach().numpy(),list(pmf[i].numpy())))
                if a.shape[0]> 1:
                    plot_distribution(data_dic,pmf_dic,i,path)
                distance_av.append(compute_prob_distance(data_dic,pmf_dic,0))
            
            plt.figure(figsize=(14, 6))
            plt.bar(np.arange(0,len(distance_av)),distance_av)  
            plt.grid()
            plt.xlabel('channels')
            plt.ylabel('JS_distance')
            pt = os.path.join(path,"dix.png") 
 
            plt.savefig(pt) 
            
            print(np.mean(np.array(distance_av)))              
                





def compute_prob_distance(data_dic,pmf_dic,dim):
    data_val = list(data_dic.values())
    pmf_val = list(pmf_dic.values()) 
    
    r = jensenshannon(data_val,pmf_val)
    if r < 0.002:
        return 0
    else: 
        return r



def plot_distribution(data_dic,pmf_dic,dim,pth):
    
    a = list(data_dic.keys())
    b = list(data_dic.values())
    aa = list(pmf_dic.keys())
    c = list(pmf_dic.values())
    tl = [str(i.item()) for i in a]  
    plt.figure(figsize=(14, 6))

    plt.bar(a,b,align='center', tick_label = tl,label = "from data")
    plt.plot(aa,c, color = "red", label = "from the net")
    plt.xlabel('quantized values')
    plt.ylabel('Frequency (%)')
    plt.grid()
    plt.locator_params(axis='y', nbins=10)
    
    #for i in range(len(b)):
    #    plt.hlines(b[i],0,a[i]) # Here you are drawing the horizontal lines
    
    path = os.path.join(pth,"channel_" + str(dim) + ".png")
    plt.legend()
    plt.savefig(path)

    plt.close()
    