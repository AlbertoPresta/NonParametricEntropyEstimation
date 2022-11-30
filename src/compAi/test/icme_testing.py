import torch 
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
#from compAi.training.sos.loss import RateDistortionLoss, RateDistorsionLossWithHemp, DifferentialFSLoss, GenericRateDistortionLoss
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
import math
import pandas as pd



model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "icme2023-factorized": FactorizedICME,
    "icme2023-hyperprior": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior

}


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()





def extract_results_on_entire_kodak(models_path,
                                    save_path,  
                                    dataloader,
                                    device = torch.device("cpu")):

    models = listdir(models_path)

    bpp_icme = []
    psnr_icme = []
    mssim_icme = []
    jensen_icme = []
    
    bpp_baseline = []
    psnr_baseline = []
    mssim_baseline = []
    jensen_baseline = []
    types = None
    
    for i,f in enumerate(models):
        if "DS_Store" in f:
            continue

        path_models = join(models_path,f)
        type_model = f.split("_")[0] #factorizedICME
        model_name = f.split(".")[0] #factorizedICME_0018
        model = load_model(models_path, model_name, type_model, device = device)
        
        #ora abbiamo il modello
        print("inizio modello ",f)
        bpp, psnr, mssim, js_distance = inference_with_arithmetic_codec(model, dataloader, device,  type_model)
        
        if "icme" in type_model:
            bpp_icme.append(bpp)
            psnr_icme.append(psnr)
            mssim_icme.append(mssim)
            jensen_icme.append(js_distance)
        else:
            bpp_baseline.append(bpp)
            psnr_baseline.append(psnr)
            mssim_baseline.append(mssim)
            jensen_baseline.append(js_distance)
        print("fine modello ",f)
    
    # ora abbiamo le lista si deve plottare    
    bpp_icme = sorted(bpp_icme)
    mssim_icme = sorted(mssim_icme)
    psnr_icme = sorted(psnr_icme)
    jensen_icme = sorted(jensen_icme)
    
    bpp_baseline = sorted(bpp_baseline)
    mssim_baseline = sorted(mssim_baseline)
    psnr_baseline = sorted(psnr_baseline)
    jensen_baseline = sorted(jensen_baseline)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #plt.figtext(.5, 0., '()', fontsize=12, ha='center')
   
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



    axes[2].plot(bpp_baseline, jensen_baseline,'-',color = 'b', label = "baseline")
    axes[2].plot(bpp_baseline, jensen_baseline,'o',color = 'b')
   
    axes[2].plot(bpp_icme, jensen_icme,'-',color = 'r', label = "EBSF")
    axes[2].plot(bpp_icme, jensen_icme,'o',color = 'r')
     
    axes[2].set_ylabel('Jensen Distance')
    axes[2].set_xlabel('Bit-rate [bpp]')
    axes[2].title.set_text('Average Jensen Distance over channels')
    axes[2].grid()
    axes[2].legend(loc='best')
    
    #if not exists(join(save_path,"metrics")):
    #    makedirs(join(save_path,"metrics"))
    
    cp =  join(save_path, "metric_comp_" + type_model + ".png")
    for ax in axes:
        ax.grid(True)
    plt.savefig(cp)
    plt.close()                        
        
    
        






def inference_with_arithmetic_codec(model, test_dataloader, device,  type_model):
    
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()
    timing_all = AverageMeter()
    timing_enc = AverageMeter()
    timing_dec = AverageMeter()
    js_distance = AverageMeter()
    start = time.time()  
    
    
    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
                        
            d = d.to(device) 
            start_all = time.time()
            
            bpp_list, probability = compute_per_channel_bpp(model, d, type_model)  
            if "icme" not in type_model:  
                out_enc = model.compress(d) # bit_stream is the compressed, output_cdf needs for decoding 
                enc_comp = time.time() - start_all
                timing_enc.update(enc_comp)
                start = time.time()  
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                timing_dec.update(time.time() - start)
                timing_all.update(time.time() - start_all)
            else:
                out_enc = model.compress_during_training(d, device = device) # bit_stream is the compressed, output_cdf needs for decoding 
                enc_comp = time.time() - start_all
                timing_enc.update(enc_comp)
                start = time.time()  
                out_dec = model.decompress_during_training(out_enc["strings"], out_enc["shape"])
                timing_dec.update(time.time() - start)
                timing_all.update(time.time() - start_all)               

            bpp= bpp_calculation(out_dec, out_enc)
            bpp_loss.update(bpp)
            psnr.update(compute_psnr(d, out_dec["x_hat"]))
            mssim.update(compute_msssim(d, out_dec["x_hat"]))  
            
            # inizio a calcolare la distanza di Jensen media su tutto il dataset kodak 
            y = model.g_a(d)
            y = y.squeeze(0)
            if "icme" in type_model:
                y = model.entropy_bottleneck.quantize(y, False)
            else:
                y = model.entropy_bottleneck.quantize(y, "symbols")
            lista_js = []
            for j in range(192):
                # extract the true probability           
                    a,b = torch.unique(y[j],return_counts=True)
                    a = a.int()
                    somma = torch.sum(b).item()
                    b = (b/somma)
                
                    if "icme" in type_model:
                        data_dic = create_dictionary(a,b)
                    else:
                        data_dic = create_dictionary(a,b)
                
                    pmf = probability[j]
                    l = int(pmf.shape[0]/2)
                    pmf_dic = dict(zip(np.arange(-l,l + 1),list(pmf.numpy())))
                
                    r =  compute_prob_distance(data_dic,pmf_dic,j)
                    lista_js.append(r)
            js_distance.update(np.mean(np.array(lista_js))) 
                
    return bpp_loss.avg, psnr.avg, mssim.avg, js_distance.avg 
    


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
    
    
    

def plot_convergence(path, savepath,mode = "baseline", lmbda = "0018"):
    
    lista_csv = [join(path,f) for f in listdir(path) if mode in f and lmbda in f]
    total_prob = np.zeros((len(lista_csv),21))
    for i in range(len(lista_csv)):
        if i%2==0 or i%2==1:
            files = [f for f in lista_csv if str(i) in f.split(".")[0][-1]][0]
            print("------------------ ",files)
            
            df = pd.read_csv(files)
            if i==0:
                x = df['x'].to_numpy()
                x = x[20:41]
            for f in list(df.columns):
                if "p_y" in f:
                    prob = df[f].to_numpy()
            
            prob = prob[20:41]
            epoch = int(files.split("_")[2][0])
            total_prob[i,:] = prob
    

 
    plt.figure(figsize=(14, 6))
    
    #plt.style.use('ggplot')
    
    plt.title('entropy model over epochs for ' + str(mode))
    plt.xlabel('x')
    plt.ylabel('probability distribution')
    plt.xticks(np.arange(-10,10))
    plt.yticks(np.arange(0,1.05,0.05))
    for i in range(total_prob.shape[0]):
    
        plt.scatter(x=x,y=total_prob[i],marker='o',label="epoch " + str((i + 1)))
        plt.plot(x,total_prob[i])
   
    plt.grid()
    plt.legend(loc='upper right')
    
    
    svp = join(savepath,"entropy_model_dix_baseline.png")
    
    plt.savefig(svp)
        



def loss_functions(lista_df):
    for p in lista_df:
        df = pd.read_csv(p)
        for st in df.columns:
            if "MIN"  in st or "MAX" in st or "step" in st:
                df.drop(st, inplace=True, axis=1)
        print("---------- ",df.columns)
        #for st in df.columns:    
        #    if "-" in st:
        #        df.columns = df.columns.str.replace(st,st.split("-")[1][6:].split("_")[0])
    

    
      



######################################################################################
######################################################################################
######################################################################################
######################################################################################
################### SINGLE IMAGE TESTING #############################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


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
    
    if not exists(join(save_path_image,"reconstruction")):
         makedirs(join(save_path_image,"reconstruction"))
        
    complete_save_path = join(save_path_image,"reconstruction",model_name + "_" + "reconstruction.png")
        
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
        

        bpp_list, probability = compute_per_channel_bpp(model, img, type_model)        
        major_ch = [i[0] for i in sorted(enumerate(bpp_list), key=lambda x:x[1])]
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(bpp_list, '.')
        ax.title.set_text('Per-channel bit-rate')
        ax.set_xlabel('Channel index')
        ax.set_ylabel('Channel bpp')
        
        if not exists(join(save_path_image,"channels_bpp")):
            makedirs(join(save_path_image,"channels_bpp"))
        
        
        complete_save_path = join(save_path_image,"channels_bpp",model_name + "_" + "channel_bpp.png")
        plt.savefig(complete_save_path)


        with torch.no_grad():
            y = model.g_a(img)
            y = y.squeeze(0)
            if "icme" in type_model:
                y = model.entropy_bottleneck.quantize(y, False)
            else:
                y = model.entropy_bottleneck.quantize(y, "symbols")
        
        #extract 5 worst probability statistics  and compare with the true one
        lista_js = []
        for i in range(192):
            # extract the true probability           
                a,b = torch.unique(y[i],return_counts=True)
                a = a.int()
                somma = torch.sum(b).item()
                b = (b/somma)
                
                if "icme" in type_model:
                    data_dic = create_dictionary(a,b)
                else:
                    data_dic = create_dictionary(a,b)
                
                pmf = probability[i]
                l = int(pmf.shape[0]/2)
                pmf_dic = dict(zip(np.arange(-l,l + 1),list(pmf.numpy())))
                
                if not exists(join(save_path_image,"latent_space",model_name)):
                    makedirs(join(save_path_image,"latent_space",model_name))
                               
                pth  = join(save_path_image,"latent_space",model_name,"channel_" + str(i) + "_distribution.png") 
                r = plot_distribution(data_dic,pmf_dic,i,pth)
                lista_js.append(r)
        print(np.mean(np.array(lista_js))) 
                      
    return bpp, mssim_val, psnr_val, np.mean(np.array(lista_js))
        



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
    

    plt.legend()
    plt.savefig(pth)

    plt.close()
    return r




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
    jensen_icme = []
    
    bpp_baseline = []
    psnr_baseline = []
    mssim_baseline = []
    jensen_baseline = []
    types = None
    
    for i,f in enumerate(models):
        if "DS_Store" in f:
            continue
        #icme2023-factorized_0018.pth
        path_models = join(models_path,f)

        type_model = f.split("_")[0] #factorizedICME
        model_name = f.split(".")[0] #factorizedICME_0018
        model = load_model(models_path, model_name, type_model, device = device)
        
        
        # compress the image
        bpp, mssim_val, psnr_val, js_val= compress_and_reconstruct_single_image(model, 
                                                                        model_name, 
                                                                        path_images, 
                                                                        image_name, 
                                                                        save_path,
                                                                        type_model)
        
        if "icme" in type_model:
            bpp_icme.append(bpp)
            mssim_icme.append(mssim_val)
            psnr_icme.append(psnr_val)
            jensen_icme.append(js_val)
            types = type_model
        else:
            bpp_baseline.append(bpp)
            mssim_baseline.append(mssim_val)
            psnr_baseline.append(psnr_val)
            jensen_baseline.append(js_val)
    
    
    complete_save_path = join(save_path,image_name)
    print("psnr baseline_      ",psnr_baseline)
    
    bpp_icme = sorted(bpp_icme)
    mssim_icme = sorted(mssim_icme)
    psnr_icme = sorted(psnr_icme)
    jensen_icme = sorted(jensen_icme)
    
    bpp_baseline = sorted(bpp_baseline)
    mssim_baseline = sorted(mssim_baseline)
    psnr_baseline = sorted(psnr_baseline)
    jensen_baseline = sorted(jensen_baseline)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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



    axes[2].plot(bpp_baseline, jensen_baseline,'-',color = 'b', label = "baseline")
    axes[2].plot(bpp_baseline, jensen_baseline,'o',color = 'b')
   
    axes[2].plot(bpp_icme, jensen_icme,'-',color = 'r', label = "EBSF")
    axes[2].plot(bpp_icme, jensen_icme,'o',color = 'r')
     
    axes[2].set_ylabel('Jensen Distance')
    axes[2].set_xlabel('Bit-rate [bpp]')
    axes[2].title.set_text('Average Jensen Distance over channels')
    axes[2].grid()
    axes[2].legend(loc='best')
    
    if not exists(join(complete_save_path,"metrics")):
        makedirs(join(complete_save_path,"metrics"))
    
    cp =  join(complete_save_path,"metrics", "metric_comp_" + type_model + ".png")
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

        else:
            y_hat, y_likelihoods = net.entropy_bottleneck(y)
            probability= extract_pmf_from_baseline(net)
        
    channel_bpps = [torch.log(y_likelihoods[0, c]).sum().item() / (-math.log(2) * num_pixels)for c in range(y.size(1))]
    return channel_bpps, probability



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


    #samples = samples_c[None, :] + pmf_start[:, None, None]

    half = float(0.5)

    lower = ep._logits_cumulative(samples - half, stop_gradient=True)
    upper = ep._logits_cumulative(samples + half, stop_gradient=True)
    sign = -torch.sign(lower + upper)
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]  
    
    
    samples = samples[:,0,:]


    return pmf


    

    
     
