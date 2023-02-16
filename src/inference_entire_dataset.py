import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from compAi.utils.parser import parse_args, ConfigParser
import argparse
from Datasets.dataset import Datasets, TestKodakDataset
from torch.utils.data import DataLoader
from compAi.test.icme_testing import extract_results_on_entire_kodak, plot_convergence
import pandas as pd
import json
import warnings
from compAi.test.icme_testing import * 
import torch
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")


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
    
    
    
    return 
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
        



def loss_functions(lista_df):

    for p in lista_df:
        if "DS_Store" in p:
            continue
        print(p)
        df = pd.read_csv(p)
        for st in df.columns:
            if "MIN"  in st or "MAX" in st or "step" in st:
                df.drop(st, inplace=True, axis=1)
        
        for st in df.columns:    
            if "-" in st:
                df.columns = df.columns.str.replace(st,st.split("-")[1][6:].split("_")[0])
        print(df.columns)
        epoch = list(df["test"])[0:20]
        if "baseline" in p:
            bpp_baseline = list(df["bpp"])[:20]
            mse_baseline = list(np.array(list(df["psnr"])[:20]))
        else:
            bpp_icme = list(df["bpp"])[:20]
            mse_icme =list(np.array(list(df["psnr"])[:20]))

    #for i in range(len(epoch)):
    #    epoch[i] += 1
    
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(16, 12))
    # make a plot
    ax.plot(epoch, mse_baseline,color="red", marker="o")
    ax.plot(epoch, mse_icme, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax.set_xlabel("epochs", fontsize = 14)
    ax.grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax.set_ylabel("psnr [dB]",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    ax2.plot(epoch,bpp_baseline,color="blue",marker="o")
    ax2.plot(epoch,bpp_icme,color="blue",marker="o",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=14)
    plt.xticks(np.arange(1,21))


    legend_elements = [Line2D([0], [0], label='Ballé2018 [5]', marker = "o"),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed')]

    ax.legend(handles=legend_elements, loc = "center right")

    # save the plot as a file
    plt.savefig('/Users/albertopresta/Desktop/icme/balle2018.pdf')
    plt.close()
    return epoch,  bpp_baseline, bpp_icme, mse_baseline, mse_icme 
    
    
    
    #fig, axes = plt.subplots(1, 2, figsize=(16, 12))
   
    #axes[0].plot(epoch, mse_baseline,color="red", marker="o",label = "Ballé2018 [5]")
    #axes[0].plot(epoch, mse_icme, color="blue", marker="o",  label = "Proposed")


    #axes[0].set_ylabel('PSRN [dB]')
    #axes[0].set_xlabel('epoch')
    #axes[0].title.set_text('PSNR comparison')
    #axes[0].grid()
    #axes[0].legend(loc='best')
        

   
    #axes[1].plot(epoch, bpp_baseline,marker="o",color = 'r', label = "Ballé2018 [5]")
    #axes[1].plot(epoch, bpp_icme,marker="o",color = 'b', label = "Proposed")
     
    #axes[1].set_ylabel('Bit-rate [bpp]')
    #axes[1].set_xlabel('epochs')
    #axes[1].title.set_text('Bit-rate comparison')
    #axes[1].grid()
    #axes[1].legend(loc='best')

    #plt.savefig('/Users/albertopresta/Desktop/icme/balle2018_2.pdf')
    #plt.close()
    




def main(config):

    

    
    device = torch.device("cpu")
    save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    models_path = "/Users/albertopresta/Desktop/icme/models/factorized/icme"
    


    import os 
    print("-----------------------------> ",os.getcwd())
    
    lista_pt = "/Users/albertopresta/Desktop/icme/files/hyperprior/plots" 
    lista_df =  [os.path.join(lista_pt, f ) for f in os.listdir(lista_pt)]
    epoch,  bpp_baseline_hype, bpp_icme_hype, mse_baseline_hype, mse_icme_hype  = loss_functions(lista_df)
    
    
    lista_pt = "/Users/albertopresta/Desktop/icme/files/factorized/plots" 
    lista_df =  [os.path.join(lista_pt, f ) for f in os.listdir(lista_pt)]
    epoch,  bpp_baseline_factorized, bpp_icme_factorized, mse_baseline_factorized, mse_icme_factorized  = loss_functions(lista_df)   
    

    lista_pt = "/Users/albertopresta/Desktop/icme/files/joint/plots" 
    lista_df =  [os.path.join(lista_pt, f ) for f in os.listdir(lista_pt)]
    epoch,  bpp_baseline_joint, bpp_icme_joint, mse_baseline_joint, mse_icme_joint  = loss_functions(lista_df)   
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
    ax[0].plot(epoch, mse_baseline_factorized,color="red", marker="o")
    ax[0].plot(epoch, mse_icme_factorized, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax[0].set_xlabel("epochs", fontsize = 14)
    ax[0].grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax[0].set_ylabel("PSNR [dB]",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax[0].twinx()
    ax2.plot(epoch,bpp_baseline_factorized, color="blue", marker="o")
    ax2.plot(epoch,bpp_icme_factorized, color="blue", marker="o",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=14)
    #plt.xticks(np.arange(1,21))
    plt.xticks(np.arange(len(epoch)), np.arange(1, len(epoch)+1))

    legend_elements = [Line2D([0], [0], label='Ballé2018 [4]', marker = "o", color = "k"),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed', color = "k")]

    ax[0].legend(handles=legend_elements, loc = "center right")
    """
    ax[1].plot(epoch,mse_baseline_hype,color="red", marker="o")
    ax[1].plot(epoch, mse_icme_hype, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax[1].set_xlabel("epochs", fontsize = 14)
    ax[1].grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax[1].set_ylabel("psnr [dB]",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax[1].twinx()
    ax2.plot(epoch,bpp_baseline_hype, color="blue", marker="o")
    ax2.plot(epoch,bpp_icme_hype, color="blue", marker="o",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=14)
    plt.xticks(np.arange(0,21))
    legend_elements = [Line2D([0], [0], label='Ballé2018 [10]', marker = "o"),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed')]
    
    ax[1].legend(handles=legend_elements, loc = "center right")
    """
    ax[1].plot(epoch,mse_baseline_joint, color="red", marker="o")
    ax[1].plot(epoch, mse_icme_joint, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax[1].set_xlabel("epochs", fontsize = 14)
    ax[1].grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax[1].set_ylabel("PSNR [dB]",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax[1].twinx()
    ax2.plot(epoch,bpp_baseline_joint, color="blue", marker="o")
    ax2.plot(epoch,bpp_icme_joint, color="blue", marker="o",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=14)
    #plt.xticks(np.arange(1,21))
    plt.xticks(np.arange(len(epoch)), np.arange(1, len(epoch)+1))
    legend_elements = [Line2D([0], [0], label='Cheng2020 [9]', marker = "o", color = "k"),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed',color = "k")]
    ax[1].legend(handles=legend_elements, loc = "center right")   
    plt.savefig('/Users/albertopresta/Desktop/icme/convergence.pdf')
    #test_dataset = TestKodakDataset(data_dir=path_images)
    #dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    #path = "/Users/albertopresta/Desktop/icme/files/factorized/convergence"
    #save_path =  "/Users/albertopresta/Desktop/icme/results/general"
    #plot_convergence(path, savepath,mode = "baseline", lmbda = "0018")
    
    #print("----------------------------------------------------------------")
    #extract_results_on_entire_kodak(models_path,    
    #                                save_path, 
    #                                dataloader,
    #                                device = torch.device("cpu"))
    #print("---------------------------------------------------------------- DONE ---------")
    


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    my_parser.add_argument("-c","--config", default="configuration/config_eval_icme.json", type=str,
                      help='config file path')
    
    args = my_parser.parse_args()
    
    config_path = args.config

    

    with open(config_path) as f:
        config = json.load(f)
    main(config)