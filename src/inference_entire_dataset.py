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
import torch
warnings.filterwarnings("ignore")



def main(config):

    

    
    device = torch.device("cpu")
    save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"
    models_path = "/Users/albertopresta/Desktop/icme/models/factorized/icme"
    


    test_dataset = TestKodakDataset(data_dir=path_images)
    dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    path = "/Users/albertopresta/Desktop/icme/files/factorized/convergence"
    savepath =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    plot_convergence(path, savepath,mode = "baseline", lmbda = "0018")
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