import math
import io
import torch
import time
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#import wandb
from collections import defaultdict
from pytorch_msssim import ms_ssim
#from ipywidgets import interact, widgets
import os
import pickle, gzip

import torchvision
import torch
import argparse
import json

import argparse
import collections


from compressai.zoo import bmshj2018_factorized



from compAi.test.evaluate import *
from compAi.test.utility import create_net_dict


from compAi.test.latent_space_utility import *


         

def main(config):
    basepath = config["basepath"]
    list_models = config["list_models"]
    list_name = config["model_name"]
    list_path = config["model_path"]
    list_test = config["test_path"]
    device = config["device"]
    save_path = config["save_path"]
    #save_path = 

    image_list = [os.path.join(list_test,f) for f in os.listdir(list_test)]
    networks = create_net_dict(list_models,list_name, list_path, basepath, dataloader =None)


    image_list = [os.path.join(list_test,f) for f in os.listdir(list_test)]
    net = networks["ufwr1"]
    #net = networks["bmshj2018-pretrained1"]
    
    samples = define_samples(net, "sos")
    pmf = net.entropy_bottleneck.pmf
    #pmf = None
    analyze_latentspace(net,image_list,sos = True, samples = samples, pmf = pmf, path =  save_path)
    
    
    
    



if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    my_parser.add_argument("-c","--config", default="configuration/config_eval.json", type=str,
                      help='config file path')
    
    args = my_parser.parse_args()
    
    config_path = args.config

    

    with open(config_path) as f:
        config = json.load(f)
    main(config)
    print("done")
