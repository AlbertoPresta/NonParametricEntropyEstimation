import yuvio 
import numpy as np 
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
import yuvio
import torch
import shutil
from torchvision.utils import save_image
import os
from Datasets.dataset import Datasets, TestKodakDataset,  VimeoDatasets
from torch.utils.data import DataLoader
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior, ICMEMeanScaleHyperprior, ICMEJointAutoregressiveHierarchicalPriors
from pytorch_msssim import ms_ssim
import math
from compressai.zoo import *
from matplotlib.lines import Line2D
path = "/Users/albertopresta/Desktop/icme/jvetoriginal/BasketballDrill_832x480_50.yuv"
savepath ="/Users/albertopresta/Desktop/icme/jvet/BasketballDrill832_480_420"


model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "minnen2019":mbt2018,
    "icme2023-factorized": FactorizedICME,
    "icme2023-hyperprior": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior,
    "icme2023-joint":ICMEJointAutoregressiveHierarchicalPriors

}

def normalize (nparray, bit_depth = 8):
    #max_value = (2**bit_depth) -1
    max_value = 2/((2**bit_depth) -1)
    #return ((nparray.astype('float32') / max_value) * 2) - 1
    return (nparray * max_value) - 1


def read_img(fname,img_size, bit_depth = 10):
    if bit_depth == 8:
        img = yuvio.mimread(fname, img_size[0], img_size[1], "yuv420p")
    else:
        img = yuvio.mimread(fname, img_size[0], img_size[1], "yuv420p10le")
        
       
    return img

import matplotlib.image

#.image.imsave(savepath + "/prova.png", image, cmap = "gray")
from os.path import join

def unroll_yuv(path,name, x,y,bd,savep):
    p = join(path,name)
    #print("porca medonna: ",p)
    img = read_img(p, (x, y),bit_depth = bd)
    
    for i in range(len(img)):
        image= img[i][0] 
        savepath = join(savep,name,"frame_" +str(i) + ".png")
        matplotlib.image.imsave(savepath, image, cmap = "gray")
    
    
    

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

class jvetdataset_png(Dataset):
    def __init__(self, data_dir,name, image_size=256):
        
        
        self.data_dir = data_dir
        self.name = name
        self.image_size = image_size
        self.path = join(self.data_dir, self.name)
        print("PATH: ",self.path)
        
        self.image_list = []
        
        frames = len(os.listdir(self.path))
        
        for i in range(frames -1):
            st = "frame_" + str(i) + ".png"
            self.image_list.append(join(self.path,st))
        
        
        
    def __getitem__(self,i):
        
        path =  self.image_list[i]        
        image = Image.open(path).convert('L')
        transform = transforms.Compose([         
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        img = transform(image)

        img = img.repeat(3,1,1)
        return img

    def __len__(self):
        return len(self.image_list)


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp


def from_state_dict(arch, state_dict):
    """Return a new model instance from `state_dict`."""
    N = state_dict["g_a.0.weight"].size(0)
    M = state_dict["g_a.6.weight"].size(0)
    if "icme" in arch:
        print("entro qua solo una volta")
        net = model_architectures[arch](N, M)


    else: 
        net = model_architectures[arch](quality = 1)
    net.load_state_dict(state_dict)
    return net


def load_pretrained_baseline(architecture, path,device = torch.device("cpu")):  

    if (architecture not in model_architectures):
        raise RuntimeError("Pre-trained model not yet available")   
    elif isfile(path) is False:
        raise RuntimeError("path is wrong!")
    mod_load = torch.load(path,map_location= device ) # extract state dictionary
    #et = net.load_state_dict(state_dict)
    net= from_state_dict(architecture, mod_load["state_dict"])
    if "icme" in path.split("/")[-1]:
        net.entropy_bottleneck.pmf = mod_load["pmf"]
        net.entropy_bottleneck.stat_pmf = mod_load["stat_pmf"]
    return net

# fasee 1 : load models 


def load_pretrained_net( mod_load, path_models, architecture, type_mode, ml):
    if "icme" in type_mode:
        N = mod_load["N"]
        M = mod_load["M"] 
        model = architecture(N = N, M = M)
        model = model.to(ml)
        model.load_state_dict(mod_load["state_dict"])  
        model.entropy_bottleneck.pmf = mod_load["pmf"]
        model.update( device = torch.device("cpu"))        
        return model
    else:
        model = load_pretrained_baseline(type_mode, path_models, device  = ml)
        model.update()
        
        return model
        
        


def load_model(path_models, name_model, type_mode, device = torch.device("cpu")):   
    if "icme" in name_model:
        print("type_model: ",type_mode)
        #complete_path = join(path_models,name_model + ".pth.tar")
        net = load_pretrained_baseline(type_mode,path_models, device  = device)
        #net.update(device = device)
    else: # use only baseline
        print("type_model: ",type_mode)
        #complete_path = join(path_models,name_model + ".pth.tar")
        net = load_pretrained_baseline(type_mode, path_models, device  = device)    
        #net.update()  
    return net




def adapt_pmf_with_momentum(net, img ,weights = [0.2, 0.8]):

    "adapt probability distribution based on image idx"
    actual_pmf = net.entropy_bottleneck.pmf  #[192,61]        
    # compute the test pmf opf the previous frame()
    x_0 = net.g_a(img[0])
    bs,ch,w,h = x_0.shape  
    x_0= x_0.reshape(ch,bs,w*h)                       
    outputs = net.entropy_bottleneck.quantize(x_0,  False)
    prob_1 = net.entropy_bottleneck._probability(outputs)   
    
    x_1 = net.g_a(img[1])
    bs,ch,w,h = x_1.shape  
    x_1= x_1.reshape(ch,bs,w*h)                       
    outputs = net.entropy_bottleneck.quantize(x_1,  False)
    prob_2 = net.entropy_bottleneck._probability(outputs)       
    res = (prob_1*weights[0] + prob_2*weights[1])/2
    return res



def adapt_pmf(net, img):

    "adapt probability distribution based on image idx"
    actual_pmf = net.entropy_bottleneck.pmf  #[192,61]        
    # compute the test pmf opf the previous frame()
    x = net.g_a(img)
    bs,ch,w,h = x.shape  
    x = x.reshape(ch,bs,w*h)                       
    outputs = net.entropy_bottleneck.quantize(x,  False)
    prob = net.entropy_bottleneck._probability(outputs)         
    res = (actual_pmf + prob)/2
    return res



def plot_results( bpp_fact, psnr_fact,bpp_base, psnr_base, name):
    fig, axes = plt.subplots(figsize=(18, 6))
   
    axes.plot(bpp_base, psnr_base,color="red",label = "Ballé2017 [8]")
    axes.plot(bpp_base, psnr_base,color="red",marker = "o")

    axes.plot( bpp_fact, psnr_fact, color="blue",  label = "Proposed")
    axes.plot( bpp_fact, psnr_fact, color="blue", marker =  "o")

    axes.set_ylabel('PSRN [dB]',fontsize = 14)
    axes.set_xlabel('bpp',fontsize = 14)
    axes.title.set_text('PSNR comparison')
    axes.grid()
    axes.legend(loc='best')
        
    name = name + "2.pdf"
    plt.savefig('/Users/albertopresta/Desktop/icme/results/jvet/' + name) 
    plt.close()
    


def plot_frame_compression_results(frames, bpp_fact, psnr_fact,bpp_base, psnr_base, name, momentum):
    
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=(18, 6))
    # make a plot
    ax.plot(frames, psnr_base,color="red", marker="o")
    ax.plot(frames, psnr_fact, color="red", marker="o", linestyle= ":")
    # set x-axis label
    ax.set_xlabel("frame", fontsize = 14)
    ax.grid(which='major', axis='x', linestyle='--')
    # set y-axis label
    ax.set_ylabel("psnr [dB]",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    ax2.plot(frames,bpp_fact,color="blue",)
    ax2.plot(frames,bpp_base,color="blue",linestyle= ":" )
    ax2.set_ylabel("Bit-rate [bpp]",color="blue",fontsize=14)
    plt.xticks(np.arange(0,21))


    legend_elements = [Line2D([0], [0], label= "Ballè2017 [8]", marker = "o",color='k'),
                        Line2D([0], [0], marker='o',linestyle= ":" , label='Proposed', color='k')]

    ax.legend(handles=legend_elements, loc = "center right",labelcolor='k')
    if momentum:
        name = name + "_momentum.pdf"
    else:
        name = name + ".pdf"
    # save the plot as a file
    plt.savefig('/Users/albertopresta/Desktop/icme/results/jvet/' + name)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
   
    axes[0].plot(frames, psnr_base,color="red",label = "Ballé2017 [8]")
    axes[0].plot(frames, psnr_base,color="red",marker  ="o")
    axes[0].plot(frames, psnr_fact, color="blue",  label = "Proposed")
    axes[0].plot(frames, psnr_fact, color="blue",  marker ="o")


    axes[0].set_ylabel('PSRN [dB]',fontsize = 14)
    axes[0].set_xlabel('frames',fontsize = 14)
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
        

   
    axes[1].plot(frames, bpp_base,  color = 'r', label = "Ballé2017 [8]")
    axes[1].plot(frames, bpp_fact,  color = 'b', label = "Proposed")
     
    axes[1].set_ylabel('Bit-rate [bpp]',fontsize = 14)
    axes[1].set_xlabel('frames',fontsize = 14)
    axes[1].title.set_text('Bit-rate comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    name = name + "2.pdf"
    plt.savefig('/Users/albertopresta/Desktop/icme/results/jvet/' + name) 
    plt.close()
    
    


def main(): 

    

    
    device = torch.device("cpu")
    save_path =  "/Users/albertopresta/Desktop/icme/results/icme/factorized/entropycode"
    path_images =  "/Users/albertopresta/Desktop/icme/kodak"

    
        
    modelpath = "/Users/albertopresta/Desktop/icme/models/models_for_jvet/joint"
    models =  listdir(modelpath)  # lista modelli da calcolare 

    test_dataset = TestKodakDataset(data_dir=path_images, names = True)
    kodak = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    test_dataset_pmf = TestKodakDataset(data_dir=path_images)
    kodak_pmf = DataLoader(dataset=test_dataset_pmf, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)

    td_path  = "/Users/albertopresta/Desktop/icme/vimeo_triplet/sequences"
    file_txt = "/Users/albertopresta/Desktop/icme/vimeo_triplet/tri_trainlist.txt"
    train_dataset = VimeoDatasets(td_path, file_txt,image_size = 256) 
    import pickle

    #b = torch.randperm(len(train_dataset)).tolist()

    with open("lista_index", "rb") as fp:   # Unpickling
        b = pickle.load(fp)

    shuffled_dataset = torch.utils.data.Subset(train_dataset, b)
    train_dataloader = DataLoader(dataset=shuffled_dataset, batch_size=1,shuffle=True,pin_memory=True,num_workers=4)

    models =  listdir(modelpath)  # lista modelli da calcolare 

    
    
    for i,(d, nm) in enumerate(kodak): 
        name = nm[0]
        if os.path.isdir(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/kodak",name)) is False:   
            os.makedirs(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/kodak",name))   
        else:
            shutil.rmtree(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/kodak",name))
            os.makedirs(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/kodak",name))  

    #bpps = {}
    #psnr = {}
    #mssim = {}

    
    bpp_total_icme = []
    psnr_total_icme = []
    
    
    bpp_total_baseline = []
    psnr_total_baseline = []
    

    transl = {"025": "18",
              "013": "22",
              "0070": "27",
              "0036":"32",
              "0018":"37",
              "0009":"42"
              }
        
                    
    total_bpp = {}
    total_psnr = {}    
    number_of_data = 24
    data_pmf = True
    lmbdas = ["0009","0018","0036","0070","013","025"]
    for lb in lmbdas:
        for f in listdir(modelpath):
            if lb not in f:
                continue 
            else:  
                print(f)
                if "DS" not in f:                   
                    path_models = join(modelpath,f) 
                    type_mode = f.split("_")[0] #factorizedICME ----> architecture
                    name_model = f.split(".")[0] #factorizedICME_0018  ----> name icme2023-joint.pth.tar 
                    checkpoint = torch.load(path_models, map_location= torch.device("cpu"))
                    architecture = model_architectures[type_mode]
                    net = load_pretrained_net(checkpoint,   path_models,architecture,  type_mode, torch.device("cpu"))   
                    if "icme" in type_mode and data_pmf is True:
                        print("entro qua")
                        res = net.define_statistical_pmf( train_dataset, device = torch.device("cpu"), idx = number_of_data) 
                        #torch.save(res, '/Users/albertopresta/Desktop/icme/files/pmf_model/' + type_mode + str(number_of_data) + 'tensor.pt')
                        print("ho finito")                 
                    #net = load_model(path_models, name_model, type_mode                    
                    for i,(d, nm) in enumerate(kodak): 
                        name = nm[0]
                        #os.makedirs(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/kodak",name))                      
                        bpp = 0
                        psnr_val = 0
                        if "icme"  not in type_mode:
                            
                            
                            #with torch.no_grad():
                            #    out_enc = net(d)
                            #    size = out_enc['x_hat'].size() 
                            #    num_pixels = size[0] * size[2] * size[3]
                            #    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_enc["likelihoods"].values())
                            #    bpp = bpp.item()
                            #    psnr_val = compute_psnr(d, out_enc["x_hat"]) 
                                
                            
                            out_enc = net.compress(d) # bit_stream is the compressed, output_cdf needs for decoding 
                            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])   
                            psnr_val = compute_psnr(d, out_dec["x_hat"]) 
                            bpp= bpp_calculation(out_dec, out_enc)                                               
                        else:                          
                            if "joint" in type_mode:
                                
                                out_enc  =  net(d, training =False)
                                size = out_enc['x_hat'].size() 
                                num_pixels = size[0] * size[2] * size[3]                               
                                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_enc["likelihoods"].values())                                
                                bpp = bpp.item()
                                psnr_val = compute_psnr(d, out_enc["x_hat"])                                 
                            else:                                
                                out_enc  = net.compress_during_training(d, device = torch.device("cpu")) # bit_stream is the compressed, output_cdf needs for decoding 
                                out_dec = net.decompress_during_training(out_enc["strings"], out_enc["shape"])  
                                psnr_val = compute_psnr(d, out_dec["x_hat"]) 
                                bpp= bpp_calculation(out_dec, out_enc)                                     
                        
                        total_bpp[type_mode] = {name: bpp}
                        total_psnr[type_mode] = {type_mode: psnr_val}
                        if "icme" in type_mode:
                            f=open("/Users/albertopresta/Desktop/icme/files/db/joint/kodak/" +  name +  "/" + str(number_of_data) +  "_prop.txt" , "a+")
                            f.write("MODE prop SEQUENCE " + name +  " QP " +  transl[lb] + " BITS " +  str(bpp) + " YPSNR " +  str(psnr_val) + "\n")
                            f.close()
                        else:

                            g=open("/Users/albertopresta/Desktop/icme/files/db/joint/kodak/" +  name  +  "/" + str(number_of_data) + "_ref.txt" , "a+") 
                            g.write("MODE ref SEQUENCE " + name +  " QP " +  transl[lb] + " BITS " +  str(bpp)+ " YPSNR " +  str(psnr_val ) +"\n")
                            g.close()
                        

# Python program to
# demonstrate merging
# of two files
    for i,(d, nm) in enumerate(kodak): 
        name = nm[0]
        data = data2 = ""
  
        # Reading data from file1
        with open("/Users/albertopresta/Desktop/icme/files/db/joint/kodak/"  +  name +  "/" + str(number_of_data) + "_prop.txt") as fp:
            data = fp.read()
  
        with open("/Users/albertopresta/Desktop/icme/files/db/joint/kodak/" +  name +  "/" + str(number_of_data) + "_ref.txt") as fp:
            data2 = fp.read()


        data += "\n"
        data += data2                     
        g=open("/Users/albertopresta/Desktop/icme/files/db/joint/total_db/total_" + str(number_of_data) + ".txt", 'a+') 
        g.write(data)
        g.close()
        # plottare i risultati!
    # dopo!!        
     
    #plot_results( bpp_total_icme, psnr_total_icme,bpp_total_baseline, psnr_total_baseline, name)

if __name__ == "__main__":
    main()
    
















