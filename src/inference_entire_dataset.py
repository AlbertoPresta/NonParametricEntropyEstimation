import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
#from compAi.utils.parser import parse_args, ConfigParser
import argparse
from Datasets.dataset import Datasets, TestKodakDataset
from torch.utils.data import DataLoader
#from compAi.test.icme_testing import extract_results_on_entire_kodak, plot_convergence
import pandas as pd
import json
import warnings
#from compAi.test.icme_testing import * 
import torch
warnings.filterwarnings("ignore")


def loss_functions(lista_df):



    for p in lista_df:
        print(p)
        df = pd.read_csv(p)
        for st in df.columns:
            if "MIN"  in st or "MAX" in st or "step" in st:
                df.drop(st, inplace=True, axis=1)

        for st in df.columns:    
            if "-" in st:
                df.columns = df.columns.str.replace(st,st.split("-")[1][6:].split("_")[0])
        epoch = list(df["test"])[:20]
        if "baseline" in p:
            bpp_baseline = list(df["bpp"])[:20]
            mse_baseline = list(np.array(list(df["mse"])[:20])*0.0018*255**2)
        else:
            bpp_icme = list(df["bpp"])[:20]
            mse_icme =list(np.array(list(df["mse"])[:20])*0.0018*255**2)

        # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(epoch, mse_baseline,color="red", marker="o")
    # set x-axis label
    ax.set_xlabel("epoch", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("mse (lambda*255**2)",color="red",fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(epoch,bpp_baseline,color="blue",marker="o")
    ax2.set_ylabel("bpp",color="blue",fontsize=14)
    fig.show()
    # save the plot as a file
    plt.savefig('C:/Users/Alberto/Desktop/icme/files/factorized/plot.png')





def main(config):

    

    
    device = torch.device("cpu")
    save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    models_path = "/Users/albertopresta/Desktop/icme/models/factorized/icme"
    


    import os 
    print("-----------------------------> ",os.getcwd())
    lista_pt = "C:/Users/Alberto/Desktop/icme/files/factorized/plots" 
    lista_df =  [os.path.join(lista_pt, f ) for f in os.listdir(lista_pt)]
    loss_functions(lista_df)
    #test_dataset = TestKodakDataset(data_dir=path_images)
    #dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    path = "/Users/albertopresta/Desktop/icme/files/factorized/convergence"
    savepath =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    #plot_convergence(path, savepath,mode = "baseline", lmbda = "0018")
    """
    print("----------------------------------------------------------------")
    extract_results_on_entire_kodak(models_path,    
                                    save_path, 
                                    dataloader,
                                    device = torch.device("cpu"))
    print("---------------------------------------------------------------- DONE ---------")
    """


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    my_parser.add_argument("-c","--config", default="configuration/config_eval_icme.json", type=str,
                      help='config file path')
    
    args = my_parser.parse_args()
    
    config_path = args.config

    

    with open(config_path) as f:
        config = json.load(f)
    main(config)