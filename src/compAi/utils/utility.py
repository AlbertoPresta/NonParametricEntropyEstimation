import shutil 
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.utils import make_grid
import time
from os.path import sep, join

from datetime import datetime
class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, config):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=config["cfg"]["trainer"]["lr"],
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=config["cfg"]["trainer"]["aux_lr"],
    )
    return optimizer, aux_optimizer


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    
    #if is_best:
    #    shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")
        

def define_quantiles(N, num_sigmoids): 
    quantiles = torch.zeros(N,1,3)
    if num_sigmoids == 1:
        init = torch.Tensor([0, 0, 1])
        quantiles = init.repeat(quantiles.size(0), 1, 1)
    elif num_sigmoids == 3:
        init = torch.Tensor([-1, 0, 2])
        quantiles = init.repeat(quantiles.size(0), 1, 1) 
    elif num_sigmoids == 7:
        init = torch.Tensor([-3, 0, 4])
        quantiles = init.repeat(quantiles.size(0), 1, 1)  
    else:
        init = torch.Tensor([-7, 0, 8])
        quantiles = init.repeat(quantiles.size(0), 1, 1)  
    return quantiles     


def plot_lilelihood(model, device, epoch):
    """Plot
    FUnction that prints and log to wandb 
    """ 
    start = time.time() 
    model.update()
    print("time to update: ",time.time() - start)
    # calculate max_length

    bottleneck = model.entropy_bottleneck
    medians = bottleneck.quantiles[:, 0, 1]
    minima = medians -  bottleneck.quantiles[:, 0, 0]
    minima = torch.ceil(minima).int()
    minima = torch.clamp(minima, min=0)
    maxima =  bottleneck.quantiles[:, 0, 2] - medians
    maxima = torch.ceil(maxima).int()
    maxima = torch.clamp(maxima, min=0)
    pmf_length = maxima + minima + 1
    max_length = pmf_length.max().item()
    
    
    log_dict = {
    "likelihood": epoch,
    "likelihood/max_length": max_length
    }
    wandb.log(log_dict)
    
      

def plot_sos(model, device, num_sigmoids, n = 1000):
    if num_sigmoids == 1:
        x_min = float((model.entropy_bottleneck.sos.b - 5 ))
        x_max = float((model.entropy_bottleneck.sos.b + 5))
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        y_values= model.entropy_bottleneck.sos(x_values.to(device))
        #plt.plot(x.tolist(), y.tolist(), 'ro', label='beta = {}'.format(model.beta), linestyle="None")
        data = [[x, y] for (x, y) in zip(x_values,y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"SoS ": wandb.plot.line(table, "x", "sos", title='SoS  with beta = {}'.format(model.entropy_bottleneck.sos.beta))})
        
        # plot inf sos 
        y_values= model.entropy_bottleneck.sos(x_values.to(device), -1)
        data_inf = [[x, y] for (x, y) in zip(x_values,y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"SoS  inf": wandb.plot.line(table_inf, "x", "sos", title='SoS  with beta = {}'.format(-1))}) 
    else:
        x_min = float((min(model.entropy_bottleneck.sos.b) + min(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.entropy_bottleneck.sos.b)+ max(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        y_values= model.entropy_bottleneck.sos(x_values.to(device))
        #plt.plot(x.tolist(), y.tolist(), 'ro', label='beta = {}'.format(model.beta), linestyle="None")
        data = [[x, y] for (x, y) in zip(x_values,y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"SoS ": wandb.plot.line(table, "x", "sos", title='SoS  with beta = {}'.format(model.entropy_bottleneck.sos.beta))})
        
        # plot inf sos 
        y_values= model.entropy_bottleneck.sos(x_values.to(device), -1)
        data_inf = [[x, y] for (x, y) in zip(x_values,y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"SoS  inf": wandb.plot.line(table_inf, "x", "sos", title='SoS  with beta = {}'.format(-1))})         
    

def plot_images(model, loader, device):
    model.eval() 
    count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            output = model(data, beta = - 1) 
            count += 1
            if count > 0:
                break
    image_input = wandb.Image(make_grid(data[0:8].cpu(), nrow=4, normalize=True), caption="input")
    image_predictions = wandb.Image(make_grid(output[0:8].cpu(), nrow=4, normalize=True), caption="reconstructions")      
    wandb.log({"test-input":image_input}, commit = False)
    wandb.log({"test-reconstructions":image_predictions}, commit = False)
    
    


def create_savepath(config):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(config["dataset"]["name"],config["cfg"]["sos"]["activation"],"pretrained",str(config["cfg"]["pretrained_entropy"]),
             str(config["cfg"]["sos"]["num_sigmoids"]),
             str(config["cfg"]["sos"]["extrema"]),config["cfg"]["annealing_procedure"]["annealing"],
             str(config["cfg"]["annealing_procedure"]["gap_factor"]),str(config["cfg"]["trainer"]["lambda"])).replace("/"."_")
    if "Plateau" in config["cfg"]["annealing_procedure"]["annealing"]:
        c = join(c,str(config["cfg"]["annealing_procedure"]["patience"]),str(config["cfg"]["annealing_procedure"]["threshold"])).replace("/","_")
    
    c_best = join(c,"best").replace("/","_")
    c = join(c,config["saving"]["suffix"]).replace("/","_")
    c_best = join(c_best,config["saving"]["suffix"]).replace("/","_")
    
    
    path = config["saving"]["filename"]
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best


