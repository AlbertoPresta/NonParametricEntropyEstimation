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
from compAi.training.baseline.loss import RateDistortionLoss
from compAi.training.baseline.step  import train_one_epoch, test_epoch
from compAi.training.baseline.utility import plot_latent_space_frequency, compress_with_ac, AverageMeter, CustomDataParallel, configure_optimizers, save_checkpoint, plot_likelihood_baseline

from compAi.utils.parser import parse_args, ConfigParser
import collections



image_models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
}







def main(config):

    wandb.define_metric("test/*", step_metric="test")
    wandb.define_metric("train/*", step_metric="train")
    wandb.define_metric("train_batch/*", step_metric="train_batch")
    wandb.define_metric("train_batch_quantiles/*", step_metric="train_batch")



    name = config["arch"]["name"]
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

    net = image_models[config["arch"]["model"]](quality = config["arch"]["quality"])

    net = net.to(device)

    if device == "cuda" and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, config)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = 20, factor = 0.5)
    criterion = RateDistortionLoss(lmbda=config["cfg"]["trainer"]["lambda"])
    print(config["cfg"]["trainer"]["lambda"])

    clip_max_norm = config["cfg"]["trainer"]["clip_max_norm"]
    last_epoch = 0
    if config["saving"]["checkpoint"]:  # load from previous checkpoint
        checkpoint = torch.load(config["checkpoint_path"], map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    counter = 0
    best_loss = float("inf")
    for epoch in range(last_epoch, config["cfg"]["trainer"]["epochs"]):
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        counter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            clip_max_norm,
            counter
        )
        
        # log on wandb train epoch result 
        

        loss = test_epoch(epoch,
                         test_dataloader,
                          net, 
                          criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        quality = config["arch"]["quality"]
        filename = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + config["saving"]["suffix"] 
        filename_best = config["saving"]["filename"] + str(config["cfg"]["trainer"]["lambda"]) + "_best" +  config["saving"]["suffix"]
    
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "N": net.N,
                    "M": net.M
            },
            is_best,
            filename, 
            filename_best
        ) 
        
  
        

        #compress_with_ac(net, test_dataloader, device,epoch)
        
        # plot sos curve 
        if epoch%10==0:
            for ii in [104,67,99,30,123,57,0,14,101,110]:
                plot_likelihood_baseline(net, device, epoch, dim = ii)
                #compress_with_ac(net, test_dataloader, device,epoch)
                plot_latent_space_frequency(net, test_dataloader, device,epoch,dim = ii, test = True)
        end = time.time()
        print("Runtime of the epoch " + str(epoch) + " is: ", end - start)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="configuration/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')



    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--ql', '--quality'], type=int, target='arch;quality'),
         CustomArgs(['--lmb', '--lambda'], type=float, target='cfg;trainer;lambda')
        


    ]
    
    wandb.init(project="jointautoregressive_icme", entity="albertopresta")
    config = ConfigParser.from_args(args, wandb.run.name, options)
    wandb.config.update(config._config)
    main(config)
    
    
