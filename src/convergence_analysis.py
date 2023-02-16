
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from os import listdir, makedirs
from compressai.zoo import *

import math
import pandas as pd
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")



def loss_functions(lista_df, f,l ):
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
        epoch = list(df["test"])[:20]
        if "baseline" in p:
            bpp_baseline = list(df["bpp"])[:20]
            mse_baseline = list(np.array(list(df["psnr"])[:20]))
        else:
            bpp_icme = list(df["bpp"])[:20]
            mse_icme =list(np.array(list(df["psnr"])[:20]))

    
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(18, 6))
    # make a plot
    ax.plot(epoch, mse_baseline,color="red", marker="o")
    ax.plot(epoch, mse_icme, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax.set_xlabel("epochs", fontsize = 30)
    ax.grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax.set_ylabel("psnr [dB]",color="red",fontsize=30)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    ax2.plot(epoch,bpp_baseline,color="blue",marker="o")
    ax2.plot(epoch,bpp_icme,color="blue",marker="o",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=30)
    plt.xticks(np.arange(0,21))


    legend_elements = [Line2D([0], [0], label= l, marker = "o",color='k'),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='our method', color='k')]

    ax.legend(handles=legend_elements, loc = "center right",labelcolor='k')
    name = l + ".pdf"
    # save the plot as a file
    plt.savefig('/Users/albertopresta/Desktop/icme/results/general/' + name)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
   
    axes[0].plot(epoch, mse_baseline,color="red", marker="o",label = l)
    axes[0].plot(epoch, mse_icme, color="blue", marker="o",  label = "Our method")


    axes[0].set_ylabel('PSRN [dB]',fontsize = 50)
    axes[0].set_xlabel('epochs',fontsize = 50)
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
        

   
    axes[1].plot(epoch, bpp_baseline,marker="o",color = 'r', label = l)
    axes[1].plot(epoch, bpp_icme,marker="o",color = 'b', label = "our method")
     
    axes[1].set_ylabel('Bit-rate [bpp]',fontsize = 50)
    axes[1].set_xlabel('epochs',fontsize = 50)
    axes[1].title.set_text('Bit-rate comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    name = l + "2.pdf"
    plt.savefig('/Users/albertopresta/Desktop/icme/results/general/' + name) 
    plt.close()
    
    return epoch, bpp_baseline, bpp_icme, mse_baseline, mse_icme
    




def plot_convergence(path, savepath,mode = "baseline", lmbda = "0018"): 
    lista_csv = [join(path,f) for f in listdir(path) if mode in f and lmbda in f]
    total_prob = np.zeros((len(lista_csv),11))
    for i in range(len(lista_csv)):
        if i%2==0 or i%2==1:
            files = [f for f in lista_csv if str(i) in f.split(".")[0][-1]][0]
            
            df = pd.read_csv(files)
            if i==0:
                x = df['x'].to_numpy()
                x = x[25:36]
            for f in list(df.columns):
                if "p_y" in f:
                    prob = df[f].to_numpy()
            
            prob = prob[25:36]
            epoch = int(files.split("_")[2][0])
            total_prob[i,:] = prob
    

 
    plt.figure(figsize=(14, 6))
    
    #plt.style.use('ggplot')

    
    plt.xlabel('x', fontsize= 20)
    plt.ylabel('probability distribution',  fontsize= 20)
    plt.xticks(np.arange(-5,5))
    plt.yticks(np.arange(0,1.05,0.05))
    for i in range(total_prob.shape[0]):
        if i == 0:
            plt.scatter(x=x,y=total_prob[i],marker='o',color = "k")
            plt.plot(x,total_prob[i], color = "k", linewidth=2,label="initial fit")
        elif i == total_prob.shape[0] -1:
            plt.scatter(x=x,y=total_prob[i],marker='o',color = "r")
            plt.plot(x,total_prob[i], color = "r", linewidth=2,label="final fit") 
        elif i == 1:
            #plt.scatter(x=x,y=total_prob[i],marker='o',label="intermediate fit")
            plt.plot(x,total_prob[i], color = "grey", linewidth=1,linestyle='dashed',label="intermediate fit" )                  
        else:
            #plt.scatter(x=x,y=total_prob[i],marker='o',label="intermediate fit")
            plt.plot(x,total_prob[i], color = "grey", linewidth=1,linestyle='dashed' )            
    plt.grid()
    plt.legend(loc='upper right')
    
    name = "entropy_model_dix_" + mode + ".pdf"
    svp = join(savepath,name)
    
    plt.savefig(svp)
        







def main(): 
    
    
    models = ["factorized","joint"]
    labels = ["Ballé2017 [3]","Minnen2018 [5]"]

    for i,f in enumerate(models):
        lista_pt = join("/Users/albertopresta/Desktop/icme/files",f,"plots")
        lab = labels[i]
        lista_df =  [join(lista_pt, f ) for f in listdir(lista_pt)]
        epoch, bpp_baseline, bpp_icme, mse_baseline, mse_icme = loss_functions(lista_df, f, lab)




    mode = ["icme"]
    
    path = "/Users/albertopresta/Desktop/icme/files/factorized/convergence"
    savepath =  "/Users/albertopresta/Desktop/icme/results/general"
    for m in mode:
        plot_convergence(path, savepath,mode = m, lmbda = "0018")
    
    
if __name__ == "__main__":
    main()