import numpy as np 
import os 
import torch.nn as nn
import torch
import wandb
from compAi.utils.AverageMeter import AverageMeter
from compAi.training.baseline.utility import compute_psnr, compute_msssim
import time


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, counter):
    model.train()
    device = next(model.parameters()).device
    
    
    
    loss_tot = AverageMeter()
    bpp = AverageMeter()
    mse = AverageMeter()
    aux = AverageMeter()
    
    timing_batch = AverageMeter()
    
    start_total = time.time()
    for i, d in enumerate(train_dataloader):
        start_batch = time.time()
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
           
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        timing_batch.update(time.time() - start_batch)

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()



        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        # log on wandb     
        wand_dict = {
            "train_batch": counter,
            "train_batch/losses_batch": out_criterion["loss"].clone().detach(),
            "train_batch/bpps_batch": out_criterion["bpp_loss"].clone().detach(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach(),
            "train_batch/aux_loss":aux_loss.clone().detach(),
            "train_batch/timing_batch":time.time() - start_batch
        }
        wandb.log(wand_dict)
        


        
        wand_dict = {
            "train_batch_quantiles":counter,
            "train_batch_quantiles/minima_0":model.entropy_bottleneck.quantiles[0,0,0],
            "train_batch_quantiles/medians_0":model.entropy_bottleneck.quantiles[0,0,1],
            "train_batch_quantiles/maxima_0":model.entropy_bottleneck.quantiles[0,0,2],
            "train_batch_quantiles/max_medians":torch.max(torch.abs(model.entropy_bottleneck.quantiles[:,0,1])),
            "train_batch_quantiles/min_minima": torch.min(model.entropy_bottleneck.quantiles[:,0,0]),
            "train_batch_quantiles/max_maxima": torch.max(model.entropy_bottleneck.quantiles[:,0,2])
        }
        
        
        wandb.log(wand_dict)
        
        counter += 1
        
        
    loss_tot.update(out_criterion["loss"].clone().detach())
    bpp.update(out_criterion["bpp_loss"].clone().detach())
    mse.update(out_criterion["mse_loss"].clone().detach())
    aux.update(aux_loss.clone().detach())
    
        
    log_dict = {
        "train":epoch,
        "train/losses_ep": loss_tot.avg,
        "train/bpps_ep": bpp.avg,
        "train/mse_ep": mse.avg,
        "train/aux_loss":aux.avg, 
        "train/average_timing":timing_batch.avg,
        "train/time_total":time.time() - start_total
    }
        
    wandb.log(log_dict)      
    
        
        
    return counter 

       


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()

    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)


            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))
            

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])

            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    log_dict = {
    "test":epoch,
    "test/loss": loss.avg,
    "test/bpp":bpp_loss.avg,
    "test/mse": mse_loss.avg, 
    "test/psnr":psnr.avg,
    "test/ssim":ssim.avg
    }
    wandb.log(log_dict)
    
    
    return loss.avg

