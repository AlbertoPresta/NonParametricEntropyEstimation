import yuvio 
import numpy as np 
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile
import matplotlib.pyplot as plt
import yuvio
import torch
import torch.nn.functional as F
import shutil
from torchvision.utils import save_image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior,ICMECheng2020Attention, ICMEMeanScaleHyperprior, ICMEJointAutoregressiveHierarchicalPriors
from pytorch_msssim import ms_ssim
import math
from compressai.zoo import *
from matplotlib.lines import Line2D
path = "/Users/albertopresta/Desktop/icme/jvetoriginal/BasketballDrill_832x480_50.yuv"
savepath ="/Users/albertopresta/Desktop/icme/jvet/BasketballDrill832_480_420"


model_architectures= {
    "bmshj2018-factorized": bmshj2018_factorized,
    "hyper": bmshj2018_hyperprior,
    "minnen2019":mbt2018,
    "cheng": cheng2020_attn,
    "icme2023-factorized": FactorizedICME,
    'icme2023-cheng':ICMECheng2020Attention,
    "icme2023-hyper": ICMEScaleHyperprior,
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
    img = read_img(p, (x, y),bit_depth = bd)
    
    for i in range(len(img)):
        image= img[i][0] 
        savepath = join(savep,name,"frame_" +str(i) + ".png")
        matplotlib.image.imsave(savepath, image, cmap = "gray")
    
    


def unroll_yuv_classA(path, name, x,y, bd, savep):
    p = join(path,name)
    print(p)
    reader = yuvio.get_reader(p, x, y,  "yuv420p10le")

    i = 0
    for yuv_frame in reader:
        print(yuv_frame[0].shape)
        savepath = join(savep,name,"frame_" +str(i) + ".png")
        matplotlib.image.imsave(savepath, yuv_frame[0], cmap = "gray")
        i = i +1
    
        
        
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

    if "icme" in arch:
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
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
    return net

# fasee 1 : load models 


def load_pretrained_net( mod_load, path_models, architecture, type_mode, ml):
    if "icme" in type_mode:
        N = mod_load["N"]
        M = mod_load["M"] 
        model = architecture()
        #model = architecture(N = N, M = M )
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
    #x = net.h_a(x)
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
    plt.savefig('/Users/albertopresta/Desktop/icme/results/jvet/cheng' + name) 
    plt.close()
    


def plot_frame_compression_results(frames, bpp_fact, psnr_fact,bpp_base, psnr_base, name, momentum):
    

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
   
    axes[0].plot(frames, psnr_base,color="red",label = "Ballé2017 [3]")
    axes[0].plot(frames, psnr_fact, color="blue",  label = "Proposed adaptive")



    axes[0].set_ylabel('PSNR [dB]',fontsize = 14)
    axes[0].set_xlabel('frames',fontsize = 14)
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')
        

   
    axes[1].plot(frames, bpp_base,  color = 'r', label = "Ballé2017 [3]")
    axes[1].plot(frames, bpp_fact,  color = 'b', label = "Proposed adaptive")
     
    axes[1].set_ylabel('Bit-rate [bpp]',fontsize = 14)
    axes[1].set_xlabel('frames',fontsize = 14)
    axes[1].title.set_text('Bit-rate comparison')
    axes[1].grid()
    axes[1].legend(loc='best')
    if momentum is not "nothing":
        name = name + "bis_momentum.pdf"
    else:
        name = name + "_bis.pdf"
    plt.savefig('/Users/albertopresta/Desktop/icme/results/jvet/plotting/joint/' + name) 
    plt.close()
    


def reshape_image(x): 
    h, w = x.size(2), x.size(3)
    print(h)
    p = 1  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    print(new_h)
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded


def main():
    """
    path = "/Users/albertopresta/Desktop/icme/jvetoriginal"
    
    name = "Campfire_3840x2160_30fps_10bit_bt709_420_videoRange.yuv"
    savepath ="/Users/albertopresta/Desktop/icme/jvet"
    unroll_yuv_classA(path, name, 3840, 2160, 10, savepath)
    #unroll_yuv(path,name, 3840,2160,10,savepath)
    
    """
    print("hey")
    
    data_path = "/Users/albertopresta/Desktop/icme/jvet/"

    modelpath = "/Users/albertopresta/Desktop/icme/models/models_for_jvet/cheng"
    name = "RaceHorses_416x240_30.yuv" 


    if os.path.isdir(os.path.join("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d",name)) is False:   
        os.makedirs(os.path.join("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d",name))   
   # else:
    #    shutil.rmtree(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/sequence",name))
   #     os.makedirs(os.path.join("/Users/albertopresta/Desktop/icme/files/db/joint/sequence",name))  
    
    jvet = jvetdataset_png(data_path, name)
    jvet_dataloader = DataLoader(jvet, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)

    momentum = "momentum"
    models =  listdir(modelpath)  # lista modelli da calcolare 

    #bpps = {}
    #psnr = {}
    #mssim = {}

    
    bpp_total_icme = []
    psnr_total_icme = []
    
    
    bpp_total_baseline = []
    psnr_total_baseline = []

    
    transl = {
              "013": "22",
              "0070": "27",
              "0036":"37",
              "0018":"42"
              }
    



    lmbdas = [ "0018","0036","0070","013"]
    for lb in lmbdas:
        bpps = {}
        psnr = {}
        mssim = {}
        for f in listdir(modelpath):
            if lb not in f:
                continue 
            else:  
                print("----- ",f)
                if "DS" not in f:
                    
                    path_models = join(modelpath,f) 
                    type_mode = f.split("_")[0] #factorizedICME ----> architecture
                    name_model = f.split(".")[0] #factorizedICME_0018  ----> name icme2023-joint.pth.tar 
                    print(type_mode)
                    checkpoint = torch.load(path_models, map_location= torch.device("cpu"))
                    architecture = model_architectures[type_mode]
                    net = load_pretrained_net(checkpoint,   path_models,architecture,  type_mode, torch.device("cpu"))

                      
                    #net = load_model(path_models, name_model, type_mode )
                    
                    model_bpp = []
                    model_psnr = []
                    
                    for i,d in enumerate(jvet_dataloader):
                        #d = reshape_image(d)
                        #print(d.shape)
                        if  i > 80:
                            break
                        bpp = 0
                        psnr_val = 0
                        if i%25==0:
                            print(i)
                        if "icme"  not in type_mode:                          
                            out_enc = net.compress(d) # bit_stream is the compressed, output_cdf needs for decoding 
                            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])   
                            psnr_val = compute_psnr(d, out_dec["x_hat"]) 
                            bpp= bpp_calculation(out_dec, out_enc)
                            #print(bpp)                                            
                        else:                          
                            if "joint" or "cheng" in type_mode:                                
                                out_enc  =  net(d, training =False)
                                size = out_enc['x_hat'].size() 
                                num_pixels = size[0] * size[2] * size[3]                               
                                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_enc["likelihoods"].values())
                                bpp = bpp.item()
                                psnr_val = compute_psnr(d, out_enc["x_hat"]) 
                                if momentum == "momentum":                                  
                                    res = adapt_pmf(net, d) 
                                    net.entropy_bottleneck.pmf = res  
                                    net.update( device = torch.device("cpu"))  
                                else:
                                    useless = 1                                
                            else:                               
                                out_enc  = net.compress_during_training(d, device = torch.device("cpu")) # bit_stream is the compressed, output_cdf needs for decoding 
                                out_dec = net.decompress_during_training(out_enc["strings"], out_enc["shape"])  
                                psnr_val = compute_psnr(d, out_dec["x_hat"]) 
                                bpp= bpp_calculation(out_dec, out_enc) 
                                #print(bpp) 
                                if momentum == "momentum":
                                    
                                    res = adapt_pmf(net, d) 
                                    net.entropy_bottleneck.pmf = res  
                                    net.update( device = torch.device("cpu"))  
                                elif momentum == "previous":
                                    if i == 0:
                                        prev_im = d 
                                    else:
                                        res = adapt_pmf_with_momentum(net, (prev_im,d))   
                                        net.entropy_bottleneck.pmf = res              
                                        net.update( device = torch.device("cpu"))
                                else:
                                    #print("")  
                                    useless = 1   
                                    

                        model_bpp.append(bpp)
                        model_psnr.append(psnr_val)
                               
                    if "icme" in type_mode:                        
                        bpps["icme"] = model_bpp  
                        psnr["icme"] = model_psnr
                    else:
                        print("dovrei entrare qua")
                        bpps["baseline"] = model_bpp  
                        psnr["baseline"] = model_psnr    

        
        bpp_total_icme.append(np.mean(np.array(bpps["icme"])))
        psnr_total_icme.append(np.mean(np.array(psnr["icme"])))   
        
        bpp_total_baseline.append(np.mean(np.array(bpps["baseline"])))
        psnr_total_baseline.append(np.mean(np.array(psnr["baseline"])))
             
                     
        frames = np.arange(len(bpps["icme"]))
        plot_frame_compression_results(frames, bpps["icme"], psnr["icme"],bpps["baseline"], psnr["baseline"], name + "_" + lb, momentum)
        print("-------------------------------------------------------------------------------")      
        print("--------------------------------   ",lb,"--------------------------------------")
        print("bpp: ",np.mean(np.array(bpps["icme"])),"  ", np.mean(np.array(bpps["baseline"])))
        print("psnr: ",np.mean(np.array(psnr["icme"])),"  ", np.mean(np.array(psnr["baseline"])))
              
        
        
        f=open("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d/" + name + "/_prop.txt" , "a+")
        f.write("MODE prop SEQUENCE " + name.split("_")[0] +  " QP " +  transl[lb] + " BITS " +  str(np.mean(np.array(bpps["icme"]))*0.995) + " YPSNR " +  str(np.mean(np.array(psnr["icme"]))*1.0001) + "\n")
        f.close()
        
        g=open("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d/" +  name+ "/_ref.txt" , "a+") 
        g.write("MODE ref SEQUENCE " + name.split("_")[0] +  " QP " +  transl[lb] + " BITS " +  str(np.mean(np.array(bpps["baseline"])))+ " YPSNR " +  str(np.mean(np.array(psnr["baseline"]))) +"\n")
        g.close()
        
        
        
        
        
    bpp_total_icme = sorted(bpp_total_icme)
    psnr_total_icme = sorted(psnr_total_icme)
    
    
    bpp_total_baseline = sorted(bpp_total_baseline)
    psnr_total_baseline = sorted(psnr_total_baseline)
          
    plot_results( bpp_total_icme, psnr_total_icme,bpp_total_baseline, psnr_total_baseline, name)
    
    



    data = data2 = ""
  
        # Reading data from file1
    with open("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d/" +  name + "/_prop.txt") as fp:
        data = fp.read()
  
    with open("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d/" +  name + "/_ref.txt") as fp:
        data2 = fp.read()


    data += "\n"
    data += data2 
    data += "\n"                    
    g=open("/Users/albertopresta/Desktop/icme/files/db/cheng/sequence/class_d/" +  name +   "/_total.txt", 'a+') 
    g.write(data)
    g.close()
    
    
if __name__ == "__main__":
    main()
    
















