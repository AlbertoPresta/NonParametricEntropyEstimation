import torch 
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



def reconstruct_images_with_encoding(networks, test_dataloader, save_path,indice, inputs_distribution):
    reconstruction = {}
    for name, net in networks.items():
        net.eval()
        with torch.no_grad():
            for i,d in enumerate(test_dataloader):
                if i == indice:
                    if "ufwr" in name or "sdf" in name or "stanh" in name or "adapter" in name:
                        start = time.time()
                        byte_stream, output_cdf, out_enc = net.compress(d, inputs_distribution = inputs_distribution)   # bit_stream is the compressed, output_cdf needs for decoding 
                        enc_time = time.time() - start
                        out_dec = net.decompress(byte_stream, output_cdf)
                        dec_time = time.time() - start
                        out_dec["x_hat"].clamp_(0.,1.)
                        bpp = compute_bpp(out_dec, out_enc, sos = False)                        
                    else:
                        start = time.time()
                        out_enc = net.compress(d)      
                        #out_enc = model.compress(x)
                        enc_time = time.time() - start
                        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                        dec_time = time.time() - start
                        bpp = compute_bpp(out_dec, out_enc, sos = False)
                    reconstruction["baseline"] = transforms.ToPILImage()(d.squeeze())
                    reconstruction[name] = transforms.ToPILImage()(out_dec['x_hat'].squeeze())  
        fix, axes = plt.subplots(5, 4, figsize=(14, 14))
        for ax in axes.ravel():
            ax.axis("off")
        

        for i, (name, rec) in enumerate(reconstruction.items()):
                #axes.ravel()[i + 1 ].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
            axes.ravel()[i ].imshow(rec)
            axes.ravel()[i].title.set_text(name)

            #plt.show()
        plt.savefig(save_path[0])
        plt.close()
                                            
                        

                    
                                    
                

def reconstruct_images_with_nn(networks, test_dataloader, save_path,indice):
    reconstruction = {}
    for name, net in networks.items():
        net.eval()
        with torch.no_grad():
            for i,d in enumerate(test_dataloader): 
                
                if i == indice:
                    
                    if "sos" in name or "sot" in name:
                        out_net, _ = net(d, "", False)
                        out_net["x_hat"].clamp_(0.,1.)
                    else:
                        out_net = net(d) 
                        out_net["x_hat"].clamp_(0.,1.)
                    reconstruction["original image"] = transforms.ToPILImage()(d.squeeze())
                    reconstruction[name] = transforms.ToPILImage()(out_net['x_hat'].squeeze())

    fix, axes = plt.subplots(5, 4, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")
    

    for i, (name, rec) in enumerate(reconstruction.items()):
            #axes.ravel()[i + 1 ].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i ].imshow(rec)
        axes.ravel()[i].title.set_text(name)

        #plt.show()
    plt.savefig(save_path[0])
    plt.close()
                  
                                                                   
                    
def plot_compression_values(networks, test_dataloader, save_path, inputs_distribution =None, entropy_estimation = True): 
    if entropy_estimation:
        
        bpp_sos, psnr_sos, mssim_sos, bpp_sot, psnr_sot, mssim_sot, bpp_baseline, psnr_baseline, mssim_baseline, bpp_icme, psnr_icme,mssim_icme =  compute_bpp_and_mse(networks, test_dataloader)
    else:
        bpp_sos, psnr_sos, mssim_sos, bpp_sot, psnr_sot, mssim_sot, bpp_baseline, psnr_baseline, mssim_baseline, bpp_icme, mssim_icme, psnr_icme  =  compute_bpp_and_mse_with_encoding(networks, test_dataloader, inputs_distribution)
    
    print("-----> bpp_icme",bpp_icme,"  ",psnr_icme,"  ",mssim_icme )
    print("-----> bpp_baseline",bpp_baseline,"  ",psnr_baseline,"  ",mssim_baseline )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')

    axes[0].plot(bpp_sos, psnr_sos,'-',color = 'r', label = "unm")
    axes[0].plot(bpp_sos, psnr_sos,'o',color = 'r')
    
    axes[0].plot(bpp_baseline, psnr_baseline,'-',color = 'b', label = "baseline")
    axes[0].plot(bpp_baseline, psnr_baseline,'o',color = 'b')
    
    axes[0].plot(bpp_sot, psnr_sot,'-',color = 'g', label = "dsf")
    axes[0].plot(bpp_sot, psnr_sot,'o',color = 'g')


    axes[0].plot(bpp_icme, psnr_icme,'-',color = 'y', label = "icme")
    axes[0].plot(bpp_icme, psnr_icme,'o',color = 'y')
    
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
    
    axes[1].plot(bpp_sos,mssim_sos,'-',color = 'r', label = "unm")
    axes[1].plot(bpp_sos,mssim_sos,'o',color = 'r')
    
    axes[1].plot(bpp_baseline,mssim_baseline,'-',color = 'b', label = "baseline")
    axes[1].plot(bpp_baseline,mssim_baseline,'o',color = 'b')

    axes[1].plot(bpp_sot,mssim_sot,'-',color = 'g', label = "dsf")
    axes[1].plot(bpp_sot,mssim_sot,'o',color = 'g')
   
    axes[1].plot(bpp_icme,mssim_icme,'-',color = 'y', label = "icme")
    axes[1].plot(bpp_icme,mssim_icme,'o',color = 'y')
    
 
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    for ax in axes:
        ax.grid(True)
    plt.savefig(save_path[2])
    plt.close()                                



def compute_bpp_and_mse_with_encoding(networks, test_dataloader, inputs_distribution):
    bpp_sos = []
    psnr_sos = []
    mssim_sos = []

    bpp_sot = []
    psnr_sot = [] 
    mssim_sot = []  



    bpp_icme = []
    psnr_icme = [] 
    mssim_icme = []  

    bpp_baseline = []
    psnr_baseline = []
    mssim_baseline = []
    for name, model in networks.items():
           
        model.eval()
        device = next(model.parameters()).device
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()

    
        mssim = torch.zeros(len(test_dataloader))
        psnr = torch.zeros(len(test_dataloader))
        with torch.no_grad():
            for i,d in enumerate(test_dataloader):
                if "ufwr" in name or "sdf" in name or "adapter" in name:
                    start = time.time()
                    byte_stream, output_cdf, out_enc = model.compress(d)   # bit_stream is the compressed, output_cdf needs for decoding                    
                    enc_time = time.time() - start
                    out_dec = model.decompress(byte_stream, output_cdf)
                    dec_time = time.time() - start
                    out_dec["x_hat"].clamp_(0.,1.)
                    bpp = compute_bpp(out_dec, out_enc, sos = True)               
                elif "icme" in name:
                    start = time.time()
                    out_enc = model.compress_during_training(d, device =  torch.device("cpu"))
                    mid = time.time()
                    print("time for encoding image ",i,": ",mid-start)
                    out_dec = model.decompress_during_training(out_enc["strings"], out_enc["shape"])
                    dec = time.time()
                    print("time for decoding image ",i,": ",dec - mid)
                    out_dec["x_hat"].clamp_(0.,1.)
                    bpp = compute_bpp(out_dec, out_enc, sos = False)                       
                else:
                    start = time.time()
                    #byte_stream, output_cdf, out_enc = model.compress(d)
                    out_enc = model.compress(d)   
                    enc_time = time.time() - start
                    #out_dec = model.decompress(byte_stream, output_cdf)
                    
                    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                    dec_time = time.time() - start   
                    out_dec["x_hat"].clamp_(0.,1.)
                    bpp = compute_bpp(out_dec, out_enc, sos = False)  
                      
                bpp_loss.update(bpp)          
                mssim[i] = -10*np.log10(1-compute_msssim(d, out_dec["x_hat"]))
                psnr[i] = compute_psnr(d, out_dec["x_hat"])
        if "ufwr" in name:
            bpp_sos.append(bpp_loss.avg)
            psnr_sos.append(torch.mean(psnr).item())
            mssim_sos.append(torch.mean(mssim).item())   
        elif "sdf"  in name or "adapter" in name :
            bpp_sot.append(bpp_loss.avg)
            psnr_sot.append(torch.mean(psnr).item())
            mssim_sot.append(torch.mean(mssim).item())
        elif "icme" in name:  
            
            bpp_icme.append(bpp_loss.avg)
            psnr_icme.append(torch.mean(psnr).item())
            mssim_icme.append(torch.mean(mssim).item())
        else:
            bpp_baseline.append(bpp_loss.avg)
            psnr_baseline.append(torch.mean(psnr).item())
            mssim_baseline.append(torch.mean(mssim).item())     
    
    print("bpp calculation for icme: ",bpp_icme)
    return  bpp_sos, psnr_sos, mssim_sos, bpp_sot, psnr_sot, mssim_sot, bpp_baseline, psnr_baseline, mssim_baseline, bpp_icme, mssim_icme, psnr_icme                         

       
import math
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)
from pytorch_msssim import ms_ssim

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp_and_mse(networks, test_dataloader):
    
    
    bpp_sos = []
    psnr_sos = []
    mssim_sos = []

    bpp_sot = []
    psnr_sot = [] 
    mssim_sot = []  


    bpp_icme = []
    psnr_icme = [] 
    mssim_icme = []  

    bpp_baseline = []
    psnr_baseline = []
    mssim_baseline = []
    for name, model in networks.items():
        
     
        model.eval()
        device = next(model.parameters()).device
        print("NAME: ",name)
        if "ufwr" in name: 
            criterion = RateDistorsionLossWithHemp(lmbda= 0.0018)
        elif "sdf" in name:
            criterion = DifferentialFSLoss(lmbda= 0.0018, wt = 0.50)
        elif "icme" in name:
            criterion = EntropyDistorsionLoss(lmbda= 0.0018)
        else:
            criterion = GenericRateDistortionLoss(lmbda= 0.0018)
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()        
        psnr = AverageMeter()
        mssim = AverageMeter()
        with torch.no_grad():
            for i,d in enumerate(test_dataloader):
                if "ufwr" in name or "sdf" in name:
                    out_net, _ = model(d, "", False)
                else:
                    out_net = model(d)
                out_criterion = criterion(out_net, d)

                bpp_loss.update(out_criterion["bpp_loss"])
                print(name,":", bpp_loss.avg.item())
            
                mssim.update(-10*np.log10(1-compute_msssim(d, out_net["x_hat"])))
                psnr.update(compute_psnr(d, out_net["x_hat"]))

    
        if "ufwr" in name:
            bpp_sos.append(bpp_loss.avg.item())
            psnr_sos.append(psnr.avg)
            mssim_sos.append(mssim.avg)   
        elif "sdf" in name:
            bpp_sot.append(bpp_loss.avg.item())
            psnr_sot.append(psnr.avg)
            mssim_sot.append(mssim.avg)
        elif "icme" in name: 
            bpp_icme.append(bpp_loss.avg.item())
            psnr_icme.append(psnr.avg)
            mssim_icme.append(mssim.avg) 
        else:
            bpp_baseline.append(bpp_loss.avg.item())
            psnr_baseline.append(psnr.avg)
            mssim_baseline.append(mssim.avg)     
    
    
    return  bpp_sos, psnr_sos, mssim_sos, bpp_sot, psnr_sot, mssim_sot, bpp_baseline, psnr_baseline, mssim_baseline , bpp_icme, psnr_icme, mssim_icme                        





import pandas as pd



def find_min(df,type): 
    t = np.array(list(df[type]))    
    if type == "bpp":
        t = t[:500]
       
    if type in ("bpp","loss"):
        return np.argmin(t)
    else:
        return np.argmax(t)
        
        


def build_plot(dict_val, type = "mssim",):
    if type not in ("bpp","mssim","loss","psnr"):
        raise ValueError("choose a valid criterion to take the epoch")   
    
    icme_bpp = []
    icme_psnr = []
    icme_mssim = []
    
    
    baseline_bpp = []
    baseline_psnr = []
    baseline_mssim = []


    list_models = list(dict_val.keys()) 
    
    for i,md in enumerate(list_models):
        
        dic = dict_val[md]
        ep = find_min(dic,type)
 
        bpp = dic.iloc[ep]["bpp"]
        mssim = dic.iloc[ep]["mssim"]
        psnr = dic.iloc[ep]["psnr"]
        print("------------    md    -------------------- ",md)

        if "icme" in md:
            print("entro in icme")
            icme_bpp.append(bpp)
            icme_psnr.append(psnr)
            icme_mssim.append(mssim)
        else:
            print("entro in baseline")
            baseline_bpp.append(bpp)
            baseline_psnr.append(psnr)
            baseline_mssim.append(mssim)           

    return sorted(baseline_bpp), sorted(baseline_psnr), sorted(baseline_mssim), sorted(icme_bpp), sorted(icme_psnr), sorted(icme_mssim)
        
        




def build_csv_dictionary(path_list):
    res = {}
    for i,p in enumerate(path_list):
        name = p.split("/")[-1][:-4] # psnr,mssim, bpp       
        df = pd.read_csv(p)
        for st in df.columns:
            if "MIN"  in st or "MAX" in st or "step" in st:
                df.drop(st, inplace=True, axis=1)
        for st in df.columns:    
            if "-" in st:
                df.columns = df.columns.str.replace(st,st.split("-")[1][6:].split("_")[0])

        for c in list(df.columns):
            mean_value=df[c].mean()
            df[c].fillna(value=mean_value, inplace=True)
        res[name] = df
    return res

import os
def plot_diagram(baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim, path, type = "psnr"): 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')



    
    axes[0].plot(baseline_bpp, baseline_psnr,'-',color = 'b', label = "factorized2018")
    axes[0].plot(baseline_bpp, baseline_psnr,'o',color = 'b')
    


    axes[0].plot(icme_bpp, icme_psnr,'-',color = 'r', label = "icme")
    axes[0].plot(icme_bpp, icme_psnr,'o',color = 'r')
    
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
    

    
    axes[1].plot(baseline_bpp, baseline_mssim,'-',color = 'b', label = "factorized2018")
    axes[1].plot(baseline_bpp, baseline_mssim,'o',color = 'b')


   
    axes[1].plot(icme_bpp, icme_mssim,'-',color = 'r', label = "icme")
    axes[1].plot(icme_bpp, icme_mssim,'o',color = 'r')
    
 
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    
    cp =  os.path.join(path,"metric_comp_ " + type + ".png")
    for ax in axes:
        ax.grid(True)
    plt.savefig(cp)
    plt.close()    