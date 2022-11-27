import argparse
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from Datasets.dataset import Datasets, TestKodakDataset, VimeoDatasets
from torch.utils.data import DataLoader
import os
from compressai.zoo import *
import wandb 
import time
from compAi.training.icme.loss import EntropyDistorsionLoss
from compAi.training.icme.step  import train_one_epoch, test_epoch
from compAi.training.icme.utility import CustomDataParallel, configure_optimizers, save_checkpoint,  plot_likelihood_baseline, plot_latent_space_frequency, plot_hyperprior_latent_space_frequency,compute_prob_distance, compress_with_ac
from compAi.models.icme import FactorizedICME, ICMEScaleHyperprior, ICMEMeanScaleHyperprior
from compAi.utils.parser import parse_args, ConfigParser
import collections



image_models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "icme2023-factorized": FactorizedICME,
    "icme2023-hyperprior": ICMEScaleHyperprior,
    "icme2023-meanscalehyperprior": ICMEMeanScaleHyperprior

}



def main(config):

    wandb.define_metric("test/*", step_metric="test")
    wandb.define_metric("train/*", step_metric="train")
    wandb.define_metric("train_batch/*", step_metric="train_batch")
    wandb.define_metric("train_batch_quantiles/*", step_metric="train_batch")


    clip_max_norm = config["cfg"]["trainer"]["clip_max_norm"]
    if config["cfg"]["seed"] is not None:
        torch.manual_seed(config["cfg"]["seed"])
        random.seed(config["cfg"]["seed"])


    
    
    print("starting datasets")
    td_path  = config["dataset"]["train_dataset"]
    file_txt = config["dataset"]["file_txt"]
    img_size = config["dataset"]["image_size"] 
    train_dataset = VimeoDatasets(td_path, file_txt,image_size = img_size) 
    batch_size = config["dataset"]["bs"]
    
    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)
    train_dataloader_plot = DataLoader(dataset=train_dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)
    print("ending datasets")
    
    
    test_dataset = TestKodakDataset(data_dir=config["dataset"]["test_dataset"])


    device = config["cfg"]["device"]



 

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config["cfg"]["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    
    model_name = config["arch"]["model"]
    N = config["arch"]["N"]
    M = config["arch"]["M"]
    lmbda = config["cfg"]["trainer"]["lambda"]
    power = config["cfg"]["trainer"]["power"]
    mode = config["cfg"]["trainer"]["mode"]
    
    if model_name in "icme2023-factorized" and mode != "factorized":
        raise ValueError(f'check loss function')
    if "hype" in model_name and mode != "hyperprior":
        raise ValueError(f'check loss function')
    
    net = image_models[model_name](N,M, power = power)

    net = net.to(device)
    print("POWER: ",net.entropy_bottleneck.power)
    if device == "cuda" and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, config)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = 20, factor = 0.5)
    

    criterion = EntropyDistorsionLoss(lmbda= lmbda, mode = mode)

    print("lambda is ",config["cfg"]["trainer"]["lambda"])
    last_epoch = 0
    if config["saving"]["checkpoint"]:  # load from previous checkpoint
        checkpoint = torch.load(config["checkpoint_path"], map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    counter = 0
    best_loss = float("inf")
    best_bpp = float("inf")
    epoch_plot = 0
    for epoch in range(last_epoch, config["cfg"]["trainer"]["epochs"]):
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        counter= train_one_epoch(net, criterion, train_dataloader, optimizer, epoch, clip_max_norm, counter)
        print("COUNTER: ",counter)
        # log on wandb train epoch result 
        
        print("calculate statistical probability model")
        if  epoch>=798: 
            net.define_statistical_pmf(train_dataloader_plot, idx = 1000)

        print("end calculating")
        loss, loss_bpp = test_epoch(epoch,test_dataloader,net, criterion)
        lr_scheduler.step(loss)
        
        print("start autoencoding")
        start_enc = time.time()


        bpp_ac = compress_with_ac(net,test_dataloader, device ,epoch)
        print("time needen for ac to encode and decode is ",time.time() - start_enc)

        is_best = loss < best_loss
        trigger = best_bpp > bpp_ac

        best_loss = min(loss, best_loss)
        best_bpp = min(loss_bpp, best_loss)
        print("actual bpp: ",loss_bpp, " against best bpp: ",best_bpp)
        
        filename = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + config["saving"]["suffix"] 
        filename_best = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + "_best" +  config["saving"]["suffix"]
    
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "stat_pmf": net.entropy_bottleneck.stat_pmf,
                    "pmf": net.entropy_bottleneck.pmf,
                    "N": net.N,
                    "M": net.M
            },
            is_best,
            filename, 
            filename_best
        ) 
        
        if trigger: 
            filename = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + "bpp" + config["saving"]["suffix"] 
            filename_best = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + "bpp" + "_best" +  config["saving"]["suffix"]
        
            save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "stat_pmf": net.entropy_bottleneck.stat_pmf,
                        "pmf": net.entropy_bottleneck.pmf,
                        "N": net.N,
                        "M": net.M
                },
                is_best,
                filename, 
                filename_best
            ) 
        if epoch%25==0 or epoch == 799:
            filename = config["saving"]["filename"] + "epoch" + str(epoch) + str(config["cfg"]["trainer"]["lambda"]) + "bpp" + config["saving"]["suffix"] 
            filename_best = config["saving"]["filename"] + "epoch" + str(epoch) +  str(config["cfg"]["trainer"]["lambda"]) + "bpp" + "_best" +  config["saving"]["suffix"]
        
            save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "stat_pmf": net.entropy_bottleneck.stat_pmf,
                        "pmf": net.entropy_bottleneck.pmf,
                        "power": net.entropy_bottleneck.power,
                        "N": net.N,
                        "M": net.M
                },
                False,
                filename, 
                filename_best
            )      
                 
        # plot sos curve 

            #plot_likelihood_baseline(net, device, epoch)
        if epoch%1==0:
            for ii in [2]:
                #(net, device, epoch,dim = ii)
                res_test = plot_latent_space_frequency(net, test_dataloader, device,epoch,dim = ii, test = True)
            """
            if epoch%10==0 :
                for ii in [0,1,2,3,55,191,127,160,172,68,100,91,87,90,88,23,10]:
                    if config["arch"]["model"] == "icme2023-factorized":
                        res_test = plot_latent_space_frequency(net, test_dataloader, device,dim = ii, test = True)
                        res_train = plot_latent_space_frequency(net, train_dataloader_plot, device,dim = ii, test = False)
                    else:
                        res_test = plot_hyperprior_latent_space_frequency(net, test_dataloader, device,dim = ii, test = True)
                        res_train = plot_hyperprior_latent_space_frequency(net, train_dataloader_plot, device,dim = ii, test = False)   
                             
            """


        """
        if epoch == 200:
            net.entropy_bottleneck.power = 10
        elif epoch == 400: #
            net.entropy_bottleneck.power = 20
        """
        end = time.time()
        print("Runtime of the epoch " + str(epoch) + " is: ", end - start)
    
    for ii in range(M):
        if config["arch"]["model"] == "icme2023-factorized":
            res_test = plot_latent_space_frequency(net, test_dataloader, device,dim = ii, test = True)
            res_train = plot_latent_space_frequency(net, train_dataloader_plot, device,dim = ii, test = False)
        else:
            res_test = plot_hyperprior_latent_space_frequency(net, test_dataloader, device,dim = ii, test = True)
            res_train = plot_hyperprior_latent_space_frequency(net, train_dataloader_plot, device,dim = ii, test = False)   


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="configuration/config_icme.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')



    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--ql', '--quality'], type=int, target='arch;quality'),
         CustomArgs(['--lmb', '--lambda'], type=float, target='cfg;trainer;lambda'),
        CustomArgs(['--pw', '--power'], type=float, target='cfg;trainer;power')


    ]
    
    wandb.init(project="analysis_latentspace", entity="albertopresta")
    config = ConfigParser.from_args(args, wandb.run.name, options)
    wandb.config.update(config._config)
    main(config)
    