import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from compAi.utils.parser import parse_args, ConfigParser
import argparse
from compAi.test.evaluate import *

from Datasets.dataset import Datasets, TestKodakDataset
from torch.utils.data import DataLoader
from compAi.test.icme_testing import * 


import warnings
warnings.filterwarnings("ignore")


import pandas as pd
def main(config):
    basepath = config["basepath"]
    list_models = config["list_models"]
    list_name = config["model_name"]
    list_path = config["model_path"]
    list_test = config["test_path"]
    
    print("list_test: ",list_test)
    
    device = config["device"]
    save_path_nn = config["save_path_nn"]
    inputs_distribution =config["inputs_distribution"]
    save_path_encoding = config["save_path_encoding"]
    ep = config["entropy_estimation"]
    td_path = "/Users/albertopresta/Desktop/hemp/flicker_2W_images"
    test_dataset = TestKodakDataset(data_dir=list_test)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    
    train_dataset = Datasets(td_path, 256) # take into consideration 
    
    train_dataset = Datasets(td_path, 256) # take into consideration 
    train_dataloader = DataLoader(train_dataset,batch_size=1,num_workers=1,shuffle=True,pin_memory=True)
    #networks = create_net_dict(list_models,list_name, list_path, basepath, dataloader = train_dataloader)


    import time
    start = time.time()


    import pandas as  pd
    import os
    #print(os.listdir("/Users/albertopresta/Desktop/hemp/files/")) 
    
    print("-------------------   ENTROPY ESTIMATION STARTING   -------------------")
    #indice = 22
    #reconstruct_images_with_nn(networks, test_dataloader, save_path_nn,indice)
    #plot_compression_values(networks, test_dataloader, save_path_nn, entropy_estimation = True)
    
    print("-------------------  ENTROPY CODING ESTIMATION STARTING  -----------------")
    #reconstruct_images_with_encoding(networks, test_dataloader, save_path_encoding,indice, inputs_distribution)
    #plot_compression_values(networks, test_dataloader, save_path_encoding, inputs_distribution = inputs_distribution , entropy_estimation = False)
    import pandas as pd
    print("time needed: ",time.time() - start)
    
    """
    path = "/Users/albertopresta/Desktop/hemp/files/hyperprior"

    total_path_list = [os.path.join(path,f) for f in os.listdir(path) if ".DS" not in f]


    c = build_csv_dictionary(total_path_list)
    baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c,type = "mssim")

    print(icme_bpp," ",icme_psnr," ",icme_mssim)
    print("----")
    print(baseline_bpp," ", baseline_psnr," ", baseline_mssim)

    path =  "/Users/albertopresta/Desktop/hemp/images/hyperprior/entropycode"
    #plot_diagram(baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim, path, type = "mssim")

    #baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c)

    #print(icme_bpp," ",icme_psnr," ",icme_mssim)


    #baseline_bpp, baseline_psnr, baseline_mssim, icme_bpp, icme_psnr, icme_mssim = build_plot(c, type = "bpp")

    #print(icme_bpp," ",icme_psnr," ",icme_mssim)
    """
    save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/kodak"
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    models_path = "/Users/albertopresta/Desktop/icme/models/factorized/icme"
    image_name = "kodim01.png"
    

    import os 
    c = "/Users/albertopresta/Desktop/icme/files/plots"
    lista_df = [os.path.join(c,j) for j in os.listdir(c)]

    loss_functions(lista_df)


    
    lista_immagini = os.listdir(path_images)
    #for f in lista_immagini:
    #    print("--------------------- ",f,"  ----------------------------")
    #    plot_diagram_and_images(models_path, save_path, path_images, f)
    
    
    
    

        







          
           


    
    
    
    



   
    
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