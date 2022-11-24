import torch 
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
from compAi.test.utility import compute_psnr, compute_msssim, compute_bpp, inference
from PIL import Image
from torchvision import transforms
import numpy as np
from compAi.training.sos.loss import RateDistortionLoss, RateDistorsionLossWithHemp, DifferentialFSLoss, GenericRateDistortionLoss
from compAi.training.icme.loss import EntropyDistorsionLoss
from compAi.utils.AverageMeter import AverageMeter
from pytorch_msssim import ms_ssim
import time
from PIL import Image
import math
from os import listdir
from functools import partial
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior, ICMEMeanScaleHyperprior
from os import makedirs
from compressai.zoo import *
from scipy.spatial.distance import jensenshannon

model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "icme2023-factorized": FactorizedICME,
    "icme2023-hyperprior": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior

}

def find_closest_bpp(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp


def pillow_encode(img, fmt='jpeg', quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp
    
    
    
  






def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        #bpp_y = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        #bpp_y = bpp #len(out_enc["strings"][0][0]) * 8.0 / num_pixels
        #bpp_z = bpp # len(out_enc["strings"][1][0]) * 8.0 / num_pixels
        #bpp = bpp_y + bpp_z
        return bpp

def compress_and_reconstruct_single_image(model,model_name, path_images, image_name,save_path, type_model, bpp_channels = True):
    """
    Args:
        model (_type_): model to be used 
        model_name: name of the model, to be used for saving the image
        
        path_images(_type_): path for the kodak folder
        image_name (_type_): name of the image 
        save_path (_type_): path for saving folder
    
    returns bpp, mssim, psnr
    """
    
    #create the folder if it not exists
    
    save_path_image = join(save_path, image_name)
    if not exists(save_path_image):
        makedirs(save_path_image)
        
    complete_save_path = join(save_path_image,model_name + "_" + "reconstruction.png")
        
    # create the path for the loading the image
    image_path = join(path_images,image_name)
    image = Image.open(image_path).convert('RGB')
    
    size = image.size     
    transform = transforms.Compose([         
            transforms.Resize((256,256)),
            transforms.ToTensor()])   
    img = transform(image)
    
    img = img.unsqueeze(0)
    
    if "icme" in type_model:
        out_enc = model.compress_during_training(img, device = torch.device("cpu"))  
        out_dec = model.decompress_during_training(out_enc["strings"], out_enc["shape"])
    else: 
        out_enc = model.compress(img)  
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    bpp = bpp_calculation(out_dec, out_enc)
   
    rec_image = transforms.ToPILImage()(out_dec['x_hat'].squeeze()) 
    
    #rec_image = rec_image.resize(size,Image.ANTIALIAS)

    mssim_val = -10*np.log10(1-compute_msssim(img, out_dec["x_hat"]))
    psnr_val = compute_psnr(img, out_dec["x_hat"])
    print("PSNR: ",psnr_val)
    print("bpp: ",bpp)
    fig, ax = plt.subplots(ncols = 1, figsize=(14, 14))

       
    ax.imshow(rec_image)
    plt.savefig(complete_save_path)
    plt.close()
    
    
    if bpp_channels is True:
        complete_save_path = join(save_path_image,model_name + "_" + "channel_bpp.png")

        bpp_list, probability, samples = compute_per_channel_bpp(model, img, type_model)        
        major_ch = [i[0] for i in sorted(enumerate(bpp_list), key=lambda x:x[1])]
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(bpp_list, '.')
        ax.title.set_text('Per-channel bit-rate')
        ax.set_xlabel('Channel index')
        ax.set_ylabel('Channel bpp')
        plt.savefig(complete_save_path)


        with torch.no_grad():
            y = model.g_a(img)
            y = y.squeeze(0)
            if "icme" in type_model:
                y = model.entropy_bottleneck.quantize(y, False)
            else:
                y = model.entropy_bottleneck.quantize(y, "symbols")
        
        #extract 5 worst probability statistics  and compare with the true one
        for i in range(5):
            # extract the true probability           
                a,b = torch.unique(y[i],return_counts=True)
                a = a.int()
                somma = torch.sum(b).item()
                b = (b/somma)
                
                if "icme" in type_model:
                    data_dic = create_dictionary(a,b)
                else:
                    data_dic = create_dictionary(a,b,samples[i].detach().numpy())
                
                pmf = probability[i]
                print(pmf)
                l = int(pmf.shape[0]/2)
                pmf_dic = dict(zip(np.arange(-l,l + 1),list(pmf.numpy())))
                
                pth  = join(save_path_image,model_name + "_" + str(i) + "_distribution.png") 
                plot_distribution(data_dic,pmf_dic,i,pth)
                        
    return bpp, mssim_val, psnr_val
        



def plot_distribution(data_dic,pmf_dic,dim,pth):
    
    a = list(data_dic.keys())
    b = list(data_dic.values())
    aa = list(pmf_dic.keys())
    c = list(pmf_dic.values())
    tl = [str(i) for i in a] 
    r = compute_prob_distance(data_dic,pmf_dic,dim) 
    plt.figure(figsize=(14, 6))

    plt.bar(a,b,align='center', tick_label = tl,label = "from data")
    plt.plot(aa,c, color = "red", label = "from the net")
    plt.xlabel('quantized values: JS is ' + str(r))
    plt.ylabel('Frequency (%)')
    plt.grid()
    plt.locator_params(axis='y', nbins=10)
    
    #for i in range(len(b)):
    #    plt.hlines(b[i],0,a[i]) # Here you are drawing the horizontal lines
    

    plt.legend()
    plt.savefig(pth)

    plt.close()




def compute_prob_distance(data_dic,pmf_dic,dim):
    data_val = list(data_dic.values())
    pmf_val = list(pmf_dic.values()) 

    
    r = jensenshannon(data_val,pmf_val)
    if r < 0.002:
        return 0
    else: 
        return r
  

def create_dictionary(a,b):
    
    t = dict(zip(a.tolist(),b.tolist()))
    print(t)
    res = {}
    levels = torch.arange(-30,31)
    for i in range(-30,31):
        if i in a:

            res[i] = t[i]
        else:
            res[i] = 0
    return res
    
    
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

def load_model(path_models, name_model, type_mode, device = torch.device("cpu"), dataloader = None):
    print("----> ",path_models)
    if "icme" in name_model:
        complete_path = join(path_models,name_model + ".pth.tar")
        print(complete_path)
        net = load_pretrained_baseline(type_mode, complete_path, device  = device)
        net.update(device = device)
    else: # use only baseline
        complete_path = join(path_models,name_model + ".pth.tar")
        net = load_pretrained_baseline(type_mode, complete_path, device  = device)    
        net.update()  
    return net

def plot_diagram_and_images(models_path, 
                            save_path, 
                            path_images, 
                            image_name, 
                            device = torch.device("cpu"), 
                            dataloader = None):
    
    models = listdir(models_path)

    bpp_icme = []
    psnr_icme = []
    mssim_icme = []
    
    bpp_baseline = []
    psnr_baseline = []
    mssim_baseline = []
    types = None
    
    for i,f in enumerate(models):
        if "DS_Store" in f:
            continue
        #icme2023-factorized_0018.pth
        path_models = join(models_path,f)

        type_model = f.split("_")[0] #factorizedICME
        model_name = f.split(".")[0] #factorizedICME_0018
        model = load_model(models_path, model_name, type_model, device = device, dataloader=dataloader)
        
        
        # compress the image
        bpp, mssim_val, psnr_val = compress_and_reconstruct_single_image(model, 
                                                                        model_name, 
                                                                        path_images, 
                                                                        image_name, 
                                                                        save_path,
                                                                        type_model)
        
        if "icme" in type_model:
            bpp_icme.append(bpp)
            mssim_icme.append(mssim_val)
            psnr_icme.append(psnr_val)
            types = type_model
        else:
            bpp_baseline.append(bpp)
            mssim_baseline.append(mssim_baseline)
            psnr_baseline.append(psnr_baseline)
    
    
    complete_save_path = join(save_path,image_name)
    
    
    bpp_icme = sorted(bpp_icme)
    mssim_icme = sorted(mssim_icme)
    psnr_icme = sorted(psnr_icme)
    
    bpp_baseline = sorted(bpp_baseline)
    mssim_baseline = sorted(mssim_baseline)
    psnr_baseline = sorted(psnr_baseline)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
   
    axes[0].plot(bpp_baseline, psnr_baseline,'-',color = 'b', label = "baseline")
    axes[0].plot(bpp_baseline, psnr_baseline,'o',color = 'b')

    axes[0].plot(bpp_icme, psnr_icme,'-',color = 'r', label = "EBSF")
    axes[0].plot(bpp_icme, psnr_icme,'o',color = 'r')
    
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
        
    axes[1].plot(bpp_baseline, mssim_baseline,'-',color = 'b', label = "baseline")
    axes[1].plot(bpp_baseline, mssim_baseline,'o',color = 'b')
   
    axes[1].plot(bpp_icme, mssim_icme,'-',color = 'r', label = "EBSF")
    axes[1].plot(bpp_icme, mssim_icme,'o',color = 'r')
     
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    
    cp =  join(complete_save_path, "metric_comp_" + type_model + ".png")
    for ax in axes:
        ax.grid(True)
    plt.savefig(cp)
    plt.close()     
    
    



def compute_per_channel_bpp(net, x, type):
    num_pixels = x.size(2) * x.size(3)
    with torch.no_grad():
        y = net.g_a(x)
        if "icme" in type:
            y_hat, y_likelihoods, probability = net.entropy_bottleneck(y)
            samples = None
        else:
            y_hat, y_likelihoods = net.entropy_bottleneck(y)
            probability, samples = extract_pmf_from_baseline(net)
        print(y.size(), y_likelihoods.size())
        
    channel_bpps = [torch.log(y_likelihoods[0, c]).sum().item() / (-math.log(2) * num_pixels)for c in range(y.size(1))]
    return channel_bpps, probability, samples



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
    samples_c = torch.arange(max_length, device=device)

    samples = samples_c[None, :] + pmf_start[:, None, None]

    half = float(0.5)

    lower = ep._logits_cumulative(samples - half, stop_gradient=True)
    upper = ep._logits_cumulative(samples + half, stop_gradient=True)
    sign = -torch.sign(lower + upper)
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]  
    return pmf, samples_c


    

    
     
