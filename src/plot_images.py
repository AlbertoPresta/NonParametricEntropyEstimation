import math
import io
import torch
from torchvision import transforms
import numpy as np
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior, ICMEMeanScaleHyperprior, ICMEJointAutoregressiveHierarchicalPriors, ICMECheng2020Attention
from pytorch_msssim import ms_ssim
from PIL import Image
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from compressai.zoo import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = 'mse'  # only pre-trained model for mse are available for now
   # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        return bpp

model_architectures = {
    "Ballé2017 [3]": FactorizedICME,
    "Ballé2018 [4]": ICMEScaleHyperprior,
    "Minnen2018 [5]":ICMEJointAutoregressiveHierarchicalPriors,
    "cheng2020 [8]":ICMECheng2020Attention,
    "Ballé2017 [3]baseline": bmshj2018_factorized,
    "Ballé2018 [4]baseline": bmshj2018_hyperprior,
    "Minnen2018 [5]baseline":mbt2018,
    "cheng2020 [8]baseline": cheng2020_attn,
}



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()
    
    
def load_pretrained_net( mod_load, architecture, type_mode, ml):

    N = mod_load["N"]
    M = mod_load["M"] 
    if "cheng" in type_mode:
        model = architecture()
    #elif "18" in type_mode:
    #    model = architecture(N = 192, M = 288 )
    else:
        model = architecture(N = N, M = M )
    model = model.to(ml)
    model.load_state_dict(mod_load["state_dict"])  
    model.entropy_bottleneck.pmf = mod_load["pmf"]
    model.update( device = torch.device("cpu"))    
   
    return model



def from_state_dict(arch, state_dict):
    net = model_architectures[arch]()
    net.load_state_dict(state_dict)
    return net


def main():


    name_imgs = ["kodim07.png"] #,"kodim11.png","kodim14.png","kodim15.png", "kodim20.png","kodim23.png"]# os.listdir("/Users/albertopresta/Desktop/icme/kodak/")#["kodim11.png","kodim15.png","kodim20.png","kodim23.png"] #os.listdir('/Users/albertopresta/Desktop/icme/kodak/')
    for name_img in name_imgs:
        print("----------- ",name_img," ------------")
        if "DS" in name_img:
            continue
        img = Image.oimg = Image.open('/Users/albertopresta/Desktop/icme/kodak/' + name_img).convert('RGB')
        #x = transforms.RandomResizedCrop(256)(img)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        print(x.shape)
        pth = "/Users/albertopresta/Desktop/icme/models/models_for_plotttin"
        
        list_models = os.listdir(pth)
        
        
        networks = {}
        
        for i,f  in enumerate(list_models):
            if "DS" not in f: 
                if "baseline" not in f:  
                                     
                    path_models = os.path.join(pth,f) 
                    type_mode = f.split(".")[0]#[:-4] #factorizedICME ----> architecture
                    #name_model = f.split(".")[0] #factorizedICME_0018  ----> name icme2023-joint.pth.tar 
                    print(type_mode)
                    checkpoint = torch.load(path_models, map_location= torch.device("cpu"))
                    #print((checkpoint["state_dict"]["g_a.8.conv_a.1.conv.4.bias"].shape))
                    architecture = model_architectures[type_mode]  
                    net = load_pretrained_net(checkpoint,    architecture,  type_mode, torch.device("cpu"))     
                    #networks[f.split(".")[0]] = net
                    networks[type_mode] = net
                else:
                    path_models = os.path.join(pth,f) 
                    type_mode = f.split(".")[0]#[:-4]
                    checkpoint = torch.load(path_models, map_location= torch.device("cpu"))
                    net = model_architectures[type_mode](quality = 1)
                    net.load_state_dict(checkpoint["state_dict"])
                    net.update(force = True)
                    #networks[f.split(".")[0]] = net 
                    networks[type_mode] = net
        
        
        outputs = {}
        bpps = {} 
        psnrs = {}
        mssms = {}
        with torch.no_grad():
            for name, net in networks.items():
                net = networks[name]
                if "baseline" not in name:
                    rv  =  net(x, training =False)
                    size = rv['x_hat'].size() 
                    num_pixels = size[0] * size[2] * size[3]  
                    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in rv["likelihoods"].values())
                    psnr_val = compute_psnr(x, rv["x_hat"])
                    outputs[name] = rv
                else: 
                    out_enc = net.compress(x) # bit_stream is the compressed, output_cdf needs for decoding 
                    out_dec = net.decompress(out_enc["strings"],out_enc["shape"])  
                    #out_dec['x_hat'].clamp_(0, 1) 
                    psnr_val = compute_psnr(x, out_dec["x_hat"]) 
                    bpp= bpp_calculation(out_dec, out_enc)
                    print(bpp)
                    
                    rv =  net(x)
                    
                    #out_dec['x_hat'].clamp_(0, 1)
                
                    size = rv['x_hat'].size() 
                    num_pixels = size[0] * size[2] * size[3]  
                     
                
                
                rv["x_hat"].clamp_(0.,1.)
                outputs[name] = rv                           


                bpps[name] = bpp
                psnrs[name] = psnr_val
                mssms[name]  = -10*np.log10(1-compute_msssim(x, rv["x_hat"]))
                
                
                


        reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
                    for name, out in outputs.items()}
        

        
        model_networks = ["Ballé2017 [3]","Ballé2017 [3]baseline","Ballé2018 [4]","Ballé2018 [4]baseline", "Minnen2018 [5]", "Minnen2018 [5]baseline","cheng2020 [8]","cheng2020 [8]baseline"]
        
        
        fix, axes = plt.subplots(4, 2, figsize=(40, 30))
        for ax in axes.ravel():
            ax.axis('off')
        

            
        for i, name in enumerate(model_networks):
            rec = reconstructions[name]
            axes.ravel()[i ].imshow(rec) # cropped for easy comparison
            if "baseline" not in name:
                
                axes.ravel()[i ].title.set_text(name + " + proposed " + "   " + "PSNR: "+ str(psnrs[name])[:4] + " " + "Bpp: " + str(bpps[name].item())[:4] )
                axes.ravel()[i ].title.set_size(22)
            else: 
                axes.ravel()[i ].title.set_text(name[:-8]  + "   " + "PSNR: "+ str(psnrs[name])[:4] + " " + "Bpp: " + str(bpps[name])[:4] )
                axes.ravel()[i ].title.set_size(22)
        #axes.ravel()[-1].imshow(img)
        #axes.ravel()[-1].title.set_text('Original')
        plt.tight_layout()
        plt.savefig("/Users/albertopresta/Desktop/icme/results/general/reconstruct/recostruction_ " + name_img)
        
        """
        bpp_total = {}
        psnr_total = {}
        mssm_total = {}
           
        for md in model_networks:
            bpp_total[md] = []
            psnr_total[md] = []
            mssm_total[md] = []
            for f in list(networks.keys()):
                if md == f[:-4]: # prendo modello specifico
                    bpp_total[md].append(bpps[f])
                    psnr_total[md].append(psnrs[f])
                    mssm_total[md].append(mssms[f])
            
        """ 
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        #plt.figtext(.5, 0., '()', fontsize=12, ha='center')
    
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]baseline"])),sorted(np.asarray(psnr_total["Ballé2017 [3]baseline"])),'-',color = 'b')
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]baseline"])),sorted(np.asarray(psnr_total["Ballé2017 [3]baseline"])),'*',color = 'b')

        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]"])), sorted(np.asarray(psnr_total["Ballé2017 [3]"])),'-',color = 'b', linestyle= ":")
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]"])), sorted(np.asarray(psnr_total["Ballé2017 [3]"])),'o',color = 'b')

        
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]baseline"])),sorted(np.asarray(psnr_total["Ballé2018 [4]baseline"])),'-',color = 'r')
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]baseline"])),sorted(np.asarray(psnr_total["Ballé2018 [4]baseline"])),'*',color = 'r')

        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]"])), sorted(np.asarray(psnr_total["Ballé2018 [4]"])),'-',color = 'r', linestyle= ":")
        axes[0].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]"])), sorted(np.asarray(psnr_total["Ballé2018 [4]"])),'o',color = 'r')

        

        axes[0].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]baseline"])),sorted(np.asarray(psnr_total["Minnen2018 [5]baseline"])),'-',color = 'g')
        axes[0].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]baseline"])),sorted(np.asarray(psnr_total["Minnen2018 [5]baseline"])),'*',color = 'g')

        axes[0].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]"])), sorted(np.asarray(psnr_total["Minnen2018 [5]"])),'-',color = 'g', linestyle= ":")
        axes[0].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]"])), sorted(np.asarray(psnr_total["Minnen2018 [5]"])),'o',color = 'g')



        axes[0].plot(sorted(np.asarray(bpp_total["cheng2020 [8]baseline"])),sorted(np.asarray(psnr_total["cheng2020 [8]baseline"])),'-',color = 'c')
        axes[0].plot(sorted(np.asarray(bpp_total["cheng2020 [8]baseline"])),sorted(np.asarray(psnr_total["cheng2020 [8]baseline"])),'*',color = 'c')

        axes[0].plot(sorted(np.asarray(bpp_total["cheng2020 [8]"])), sorted(np.asarray(psnr_total["cheng2020 [8]"])),'-',color = 'c', linestyle= ":")
        axes[0].plot(sorted(np.asarray(bpp_total["cheng2020 [8]"])), sorted(np.asarray(psnr_total["cheng2020 [8]"])),'o',color = 'c')

        
        
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
    
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]baseline"])),sorted(np.asarray(mssm_total["Ballé2017 [3]baseline"])),'-',color = 'b')
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]baseline"])),sorted(np.asarray(mssm_total["Ballé2017 [3]baseline"])),'*',color = 'b')

        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]"])), sorted(np.asarray(mssm_total["Ballé2017 [3]"])),'-',color = 'b', linestyle= ":")
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2017 [3]"])), sorted(np.asarray(mssm_total["Ballé2017 [3]"])),'o',color = 'b')

        
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]baseline"])),sorted(np.asarray(mssm_total["Ballé2018 [4]baseline"])),'-',color = 'r')
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]baseline"])),sorted(np.asarray(mssm_total["Ballé2018 [4]baseline"])),'*',color = 'r')

        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]"])), sorted(np.asarray(mssm_total["Ballé2018 [4]"])),'-',color = 'r', linestyle= ":")
        axes[1].plot(sorted(np.asarray(bpp_total["Ballé2018 [4]"])), sorted(np.asarray(mssm_total["Ballé2018 [4]"])),'o',color = 'r')

        axes[1].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]baseline"])),sorted(np.asarray(mssm_total["Minnen2018 [5]baseline"])),'-',color = 'g')
        axes[1].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]baseline"])),sorted(np.asarray(mssm_total["Minnen2018 [5]baseline"])),'*',color = 'g')

        axes[1].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]"])), sorted(np.asarray(mssm_total["Minnen2018 [5]"])),'-',color = 'g', linestyle= ":")
        axes[1].plot(sorted(np.asarray(bpp_total["Minnen2018 [5]"])), sorted(np.asarray(mssm_total["Minnen2018 [5]"])),'o',color = 'g')



        axes[1].plot(sorted(np.asarray(bpp_total["cheng2020 [8]baseline"])),sorted(np.asarray(mssm_total["cheng2020 [8]baseline"])),'-',color = 'c')
        axes[1].plot(sorted(np.asarray(bpp_total["cheng2020 [8]baseline"])),sorted(np.asarray(mssm_total["cheng2020 [8]baseline"])),'*',color = 'c')

        axes[1].plot(sorted(np.asarray(bpp_total["cheng2020 [8]"])), sorted(np.asarray(mssm_total["cheng2020 [8]"])),'-',color = 'c', linestyle= ":")
        axes[1].plot(sorted(np.asarray(bpp_total["cheng2020 [8]"])), sorted(np.asarray(mssm_total["cheng2020 [8]"])),'o',color = 'c')

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
        plt.savefig("/Users/albertopresta/Desktop/icme/results/general/kodak/RD_total_" + name_img)
        plt.close()    
        """ 
        
    """
    plt.subplots(figsize=(16, 12)) 
    axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
    axes.ravel()[0].title.set_text('Original')
    plt.savefig("/Users/albertopresta/Desktop/icme/results/general/original_ " + name_img)
    """

if __name__ == "__main__":
    main()
    



