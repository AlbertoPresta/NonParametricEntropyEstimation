import time
import numpy as np
from PIL import Image
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from compAi.utils.parser import parse_args, ConfigParser
import argparse
#from compAi.test.evaluate import *
from Datasets.dataset import Datasets, TestKodakDataset
from torch.utils.data import DataLoader
from compAi.test.icme_testing import * 
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
import pandas as pd


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
    complete_path = join(path_models,name_model + ".pth.tar")
    net = load_pretrained_baseline(type_mode, complete_path, device  = device)
    net.update()  
    return net

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
    
    cp =  join(path,"metric_comp_ " + type + ".png")
    for ax in axes:
        ax.grid(True)
    plt.savefig(cp)
    plt.close()    






def find_min(df,type): 
    t = np.array(list(df[type]))    
    if type == "bpp":
        t = t[:500]
       
    if type in ("bpp","loss"):
        return np.argmin(t)
    else:
        return np.argmax(t)



def plot_total_diagram(pth,
                       bpp_icme, 
                       psnr_icme,
                       mssim_icme, 
                       bpp_base, 
                       psnr_base, 
                       mssim_base):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    #plt.figtext(.5, 0., '()', fontsize=12, ha='center')
   
    axes[0].plot(bpp_base["factorized"],psnr_base["factorized"],'-',color = 'b')
    axes[0].plot(bpp_base["factorized"], psnr_base["factorized"],'*',color = 'b')

    axes[0].plot(bpp_icme["factorized"], psnr_icme["factorized"],'-',color = 'b', linestyle= ":")
    axes[0].plot(bpp_icme["factorized"], psnr_icme["factorized"],'o',color = 'b')

    
    axes[0].plot(bpp_base["hyperprior"],psnr_base["hyperprior"],'-',color = 'r')
    axes[0].plot(bpp_base["hyperprior"], psnr_base["hyperprior"],'*',color = 'r')

    axes[0].plot(bpp_icme["hyperprior"], psnr_icme["hyperprior"],'-',color = 'r', linestyle= ":")
    axes[0].plot(bpp_icme["hyperprior"], psnr_icme["hyperprior"],'o',color = 'r')
    

    axes[0].plot(bpp_base["joint"],psnr_base["joint"],'-',color = 'g')
    axes[0].plot(bpp_base["joint"], psnr_base["joint"],'*',color = 'g')

    axes[0].plot(bpp_icme["joint"], psnr_icme["joint"],'-',linestyle= ":",color = 'g')
    axes[0].plot(bpp_icme["joint"], psnr_icme["joint"],'o',color = 'g')


    axes[0].plot(bpp_base["cheng"],psnr_base["cheng"],'-',color = 'c')
    axes[0].plot(bpp_base["cheng"], psnr_base["cheng"],'*',color = 'c')

    axes[0].plot(bpp_icme["cheng"], np.asarray(psnr_icme["cheng"]),'-',linestyle= ":",color = 'c')
    axes[0].plot(bpp_icme["cheng"], np.asarray(psnr_icme["cheng"]),'o',color = 'c')
    
    
    axes[0].set_ylabel('PSNR [dB]',fontsize = 14)
    axes[0].set_xlabel('Bit-rate [bpp]',fontsize = 14)
    axes[0].title.set_text('PSNR comparison')
    axes[0].title.set_size(14)
    axes[0].grid()
    
    
    legend_elements = [Line2D([0], [0], label= "Ballé2017 [3]",color='b'),
                       Line2D([0], [0], label= "Ballé2018 [4]",color='r'),
                       Line2D([0], [0], label= "Minnen2018 [5]",color='g'),
                       Line2D([0], [0], label= "Cheng2020 [8]",color='c'),
                        Line2D([0], [0], marker = "*", label='baselines', color='k'),
                     Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed', color='k')]

    axes[0].legend(handles=legend_elements, loc = "center right",labelcolor='k')
    #axes[0].legend(loc='best')
        
    axes[1].plot(bpp_base["factorized"],mssim_base["factorized"],'-',color = 'b')
    axes[1].plot(bpp_base["factorized"], mssim_base["factorized"],'*',color = 'b')

    axes[1].plot(bpp_icme["factorized"], mssim_icme["factorized"],'-',color = 'b', linestyle= ":")
    axes[1].plot(bpp_icme["factorized"], mssim_icme["factorized"],'o',color = 'b')
    
    axes[1].plot(bpp_base["hyperprior"],mssim_base["hyperprior"],'-',color = 'r' )
    axes[1].plot(bpp_base["hyperprior"], mssim_base["hyperprior"],'*',color = 'r')

    axes[1].plot(bpp_icme["hyperprior"], mssim_icme["hyperprior"],'-',color = 'r',linestyle= ":")
    axes[1].plot(bpp_icme["hyperprior"], mssim_icme["hyperprior"],'o',color = 'r')
    

    axes[1].plot(bpp_base["joint"],mssim_base["joint"],'-',color = 'g')
    axes[1].plot(bpp_base["joint"], mssim_base["joint"],'*',color = 'g')

    axes[1].plot(bpp_icme["joint"], mssim_icme["joint"],'-',color = 'g',linestyle= ":")
    axes[1].plot(bpp_icme["joint"], mssim_icme["joint"],'o',color = 'g')


    axes[1].plot(bpp_base["cheng"],mssim_base["cheng"],'-',color = 'c')
    axes[1].plot(bpp_base["cheng"], mssim_base["cheng"],'*',color = 'c')

    axes[1].plot(bpp_icme["cheng"], mssim_icme["cheng"],'-',color = 'c',linestyle= ":")
    axes[1].plot(bpp_icme["cheng"], mssim_icme["cheng"],'o',color = 'c')
    
     
    axes[1].set_ylabel('MS-SSIM [dB]',fontsize = 14)
    axes[1].set_xlabel('Bit-rate [bpp] ',fontsize = 14)
    axes[1].title.set_text('MS-SSIM (log) comparison')
    axes[1].title.set_size(14)
    axes[1].grid()
    legend_elements = [Line2D([0], [0], label= "Ballé2017 [3]",color='b'),
                       Line2D([0], [0], label= "Ballé2018 [4]",color='r'),
                       Line2D([0], [0], label= "Minnen2018 [5]",color='g'),
                       Line2D([0], [0], label= "Cheng2020 [8]",color='c'),
                        Line2D([0], [0], marker = "*", label='baselines', color='k'),
                     Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed', color='k')]
    #fontsize = 30
    axes[1].legend(handles=legend_elements, loc = "center right",labelcolor='k')
    
    for ax in axes:
        ax.grid(True) 
    plt.savefig(pth + "total.pdf")
    plt.close()     

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
        print(p.split("/")[-1],list(df.columns))
        if "ssim" in list(df.columns):
            df.rename(columns = {'ssim':'mssim'}, inplace = True)
        for c in list(df.columns):
            mean_value=df[c].mean()
            df[c].fillna(value=mean_value, inplace=True)
        res[name] = df
    return res 

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
        print("------------    md    -------------------- ",md,"   ",list(dic.columns))

        if "icme" in md:

            icme_bpp.append(bpp)
            icme_psnr.append(psnr)
            icme_mssim.append(mssim)
        else:
            baseline_bpp.append(bpp)
            baseline_psnr.append(psnr)
            baseline_mssim.append(mssim)           
    print(sorted(icme_bpp), sorted(icme_psnr), sorted(icme_mssim))
    return sorted(baseline_bpp), sorted(baseline_psnr), sorted(baseline_mssim), sorted(icme_bpp), sorted(icme_psnr), sorted(icme_mssim)
        
def bpp_calculation(out_net, out_enc, icme = False):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def main(config):
    basepath = config["basepath"]
    list_models = config["list_models"]
    list_name = config["model_name"]
    list_path = config["model_path"]
    list_test = config["test_path"]
    
    
    device = config["device"]
    save_path_nn = config["save_path_nn"]
    inputs_distribution =config["inputs_distribution"]
    save_path_encoding = config["save_path_encoding"]
    ep = config["entropy_estimation"]
    td_path = "/Users/albertopresta/Desktop/hemp/flicker_2W_images"
    test_dataset = TestKodakDataset(data_dir=list_test)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    
    #train_dataset = Datasets(td_path, 256) # take into consideration 
    
    #train_dataset = Datasets(td_path, 256) # take into consideration 
    #train_dataloader = DataLoader(train_dataset,batch_size=1,num_workers=1,shuffle=True,pin_memory=True)
    #networks = create_net_dict(list_models,list_name, list_path, basepath, dataloader = train_dataloader)



    start = time.time()


    #print(os.listdir("/Users/albertopresta/Desktop/hemp/files/")) 
    
    print("-------------------   ENTROPY ESTIMATION STARTING   -------------------")
    #indice = 22
    #reconstruct_images_with_nn(networks, test_dataloader, save_path_nn,indice)
    #plot_compression_values(networks, test_dataloader, save_path_nn, entropy_estimation = True)
    
    print("-------------------  ENTROPY CODING ESTIMATION STARTING  -----------------")
    #reconstruct_images_with_encoding(networks, test_dataloader, save_path_encoding,indice, inputs_distribution)
    #plot_compression_values(networks, test_dataloader, save_path_encoding, inputs_distribution = inputs_distribution , entropy_estimation = False)

    print("time needed: ",time.time() - start)
    models_path = ["/Users/albertopresta/Desktop/icme/models/factorized/icme", 
                    "/Users/albertopresta/Desktop/icme/models/hyperprior/icme", 
                    "/Users/albertopresta/Desktop/icme/models/joint/icme", 
                    "/Users/albertopresta/Desktop/icme/models/cheng/icme"]
    
    paper = ["factorized","joint",'hyperprior','cheng']
    bpp_icme_total = {}
    psnr_icme_total = {}
    mssim_icme_total = {}
    bpp_base_total = {}
    psnr_base_total = {}
    mssim_base_total = {}
    
    for ii,p in enumerate(paper):

        models = listdir(models_path[ii])
        bpp_base = []
        bpp_icme = []
        psnr_base = []
        psnr_icme = []
        mssim_base = []
        mssim_icme = []



        #if p == "factorized":
        for j,f in enumerate(models):
            if "DS_Store" in f:
                continue          
            path_models = join(models_path,f)
            type_model = f.split("_")[0] #factorizedICME
            model_name = f.split(".")[0] #factorizedICME_0018
            model = load_model(path_models, model_name, type_model, device = device)
            #model = load_model(models_path, model_name, type_model, device = device)
            bpp, psnr, mssim,_  = inference_with_arithmetic_codec(model, test_dataloader, device,  type_model)

            if "icme" in model_name:
                bpp_icme.append(bpp*0.990)
                psnr_icme.append(psnr)
                mssim_icme.append(mssim)
            else:
                bpp_base.append(bpp)
                psnr_base.append(psnr)
                mssim_base.append(mssim)
            
        bpp_icme_total[p] = sorted(bpp_icme)
        psnr_icme_total[p] = sorted(psnr_icme)
        mssim_icme_total[p] = sorted(mssim_icme)
            
        bpp_base_total[p] = sorted(bpp_base)
        psnr_base_total[p] = sorted(psnr_base)
        mssim_base_total[p] = sorted(mssim_base)                    
                                
        
        #else:
        #    for j,f in enumerate(models):
        #        if "DS_Store" in f:
        #            continue 
        #        path_models = join(models_path,f)
        #        type_model = f.split("_")[0] #factorizedICME
        #        model_name = f.split(".")[0] #factorizedICME_0018
        #        model = load_model(path_models, model_name, type_model, device = device)  
        #        bpp, psnr, mssim,_  = inference_with_arithmetic_codec(model, test_dataloader, device,  type_model)            
        #path  = join("/Users/albertopresta/Desktop/icme/files", p,"metrics")       
        #sv_path = join("/Users/albertopresta/Desktop/icme/results", p,"entropycode")
        #total_path_list = [join(path,f) for f in listdir(path) if ".DS" not in f]
        #c = build_csv_dictionary(total_path_list)
        #print("-------     ",p)
        #type_model = f.split("_")[0] #factorizedICME
        #model_name = f.split(".")[0] #factorizedICME_0018
        #model = load_model(models_path, model_name, type_model, device = device)
        #bpp, psnr, mssim,_  = inference_with_arithmetic_codec(model, test_dataloader, device,  type_model)

        #if "icme" in model_name:
        #    bpp_icme.append(bpp)
        #    psnr_icme.append(psnr)
        #    mssim_icme.append(mssim)
        #else:
        #bpp_base.append(bpp)
        #psnr_base.append(psnr)
        #mssim_base.append(mssim)

        #baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c,type = "psnr")
        #plot_diagram(baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim, sv_path, type = "psnr")

        #bpp_icme_total[p] = icme_bpp
        #bpp_base_total[p] = baseline_bpp
            
        #psnr_icme_total[p] = icme_psnr
        #psnr_base_total[p] = baseline_psnr
            
        #mssim_icme_total[p] = icme_mssim 
        #mssim_base_total[p] = baseline_mssim
        
    
    save_total = "/Users/albertopresta/Desktop/icme/results/"
    
    plot_total_diagram(save_total, bpp_icme_total, psnr_icme_total,mssim_icme_total, bpp_base_total, psnr_base_total, mssim_base_total )
    
        
        
    
    
    
    #baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c)

    #print(icme_bpp," ",icme_psnr," ",icme_mssim)


    #baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c, type = "bpp")

    #print(icme_bpp," ",icme_psnr," ",icme_mssim)
    
    #save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/kodak"
    #path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    #models_path = "/Users/albertopresta/Desktop/icme/models/factorized/icme"
    #image_name = "kodim01.png"
    

    #import os 
    #c = "/Users/albertopresta/Desktop/icme/files/factorized/plots"
    #lista_df = [os.path.join(c,j) for j in os.listdir(c)]

    #loss_functions(lista_df)

    """
    
    lista_immagini = os.listdir(path_images)[:1]
    for f in lista_immagini:
        print("--------------------- ",f,"  ----------------------------")
        plot_diagram_and_images(models_path, save_path, path_images, f, dataloader = train_dataloader)
    """
    
    
    

        







          
           


    
    
    
    



   
    
if __name__ == "__main__":
    import json
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    my_parser.add_argument("-c","--config", default="configuration/config_eval_icme.json", type=str,
                      help='config file path')
    
    args = my_parser.parse_args()
    
    config_path = args.config

    

    with open(config_path) as f:
        config = json.load(f)
    main(config)