import numpy as np 
import torch
import wandb
from pytorch_msssim import ms_ssim
from compAi.utils.AverageMeter import AverageMeter
import math 
import time 
def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def train_one_epoch( model, criterion, train_dataloader, optimizer,epoch,clip_max_norm, counter):

    model.train()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    entropy_loss = AverageMeter()
    timing_batch = AverageMeter()
    #aux = AverageMeter()



    start_total = time.time()
    for i, d in enumerate(train_dataloader):
        start_batch = time.time()
        counter += 1
        d = d.to(device)
        optimizer.zero_grad()
        out_net = model(d, training = True) 

        #aux_loss = model.aux_loss()
        #aux_loss.backward()
        #aux_optimizer.step()

        out_criterion = criterion(out_net, d, ep = epoch)
        out_criterion["loss"].backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) 
        optimizer.step()
               
        
        timing_batch.update(time.time()- start_batch)
        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())
        entropy_loss.update(out_criterion["entropy"].clone().detach())
        #aux.update(aux_loss.clone().detach())



           
        #with torch.no_grad():
        #    model.clamp_weight(0) # mettere zero è vantaggioso? per noi si 

        if i % 50 == 0:

            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.6f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():6f} |'
                f'\tentropy loss: {out_criterion["entropy"].item():6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                )
        # log on wandb

        if np.isnan(out_criterion["loss"].item()):
            print(i, "loss is nan")
        if np.isnan(out_criterion["bpp_loss"].item()):
            print(i,"bpp loss is nan")
        if np.isnan(out_criterion["mse_loss"].item()):
            print(i,"mse loss is nan")

             
        wand_dict = {
            "train_batch": counter,
            "train_batch/losses_batch": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
            "train_batch/entropy":out_criterion["entropy"].clone().detach().item(),
            "train_batch/power":model.entropy_bottleneck.power,
            "train_batch/timing_batch":time.time()- start_batch
        }
        wandb.log(wand_dict)
                
        # we have to augment beta here 
    

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/entropy":entropy_loss.avg,
        "train/average_timing":timing_batch.avg,
        "train/time_total":time.time() - start_total
       
        }
        
    wandb.log(log_dict)
    return  counter



def test_epoch(epoch, test_dataloader, model, criterion  ):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()

    hype_loss = AverageMeter()
    gauss_loss = AverageMeter()

    mse_loss = AverageMeter()
    entropy_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()


    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            
            d = d.to(device)
            out_net = model(d, training =False) # counter = 0
            out_criterion = criterion(out_net, d)

            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))           
            bpp_loss.update(out_criterion["bpp_loss"].clone().detach())

            hype_loss.update(out_criterion["bpp_hype"].clone().detach())
            gauss_loss.update(out_criterion["bpp_gauss"].clone().detach())
            loss.update(out_criterion["loss"].clone().detach())
            mse_loss.update(out_criterion["mse_loss"].clone().detach())
            entropy_loss.update(out_criterion["entropy"].clone().detach())
 

            
          


    if i%1==0:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.6f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.6f} |"
            f"\tentropy: {entropy_loss.avg:.6f}\n"
        )

    log_dict = {
    "test":epoch,
    "test/loss": loss.avg,
    "test/bpp":bpp_loss.avg,
    "test/bpp_hype":hype_loss.avg,
    "test/bpp_gauss":gauss_loss.avg,
    "test/mse": mse_loss.avg,
    "test/entropy":entropy_loss.avg,
    "test/psnr":psnr.avg,
    "test/ssim":ssim.avg,
    }
    wandb.log(log_dict) 
    return loss.avg, bpp_loss.avg
