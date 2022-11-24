import torch
import wandb
import math

from pytorch_msssim import ms_ssim




def compute_per_channel_bpp( test_dataloader, model,  sc_type, hype = True):
    model.eval()
    device = next(model.parameters()).device
    res = torch.zeros((model.entropy_bottleneck.channels, len(test_dataloader)))
    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            num_pixels = d.size(2) * d.size(3)
            d = d.to(device)         
            y = model.g_a(d)
            if hype:
                y = model.h_a(y)
            _, y_likelihoods ,_, _ = model.entropy_bottleneck(y,  sc_type, False)
            channel_bpps = [torch.log(y_likelihoods[0, c]).sum().item() / (-math.log(2) * num_pixels) for c in range(y.size(1))]
            channel_bpps = torch.tensor(channel_bpps)
            res[:,i] = channel_bpps
    channels_bpps = res.mean(dim = 1)
    
       
    x_values = torch.arange(model.N)

    data = [[x, y] for (x, y) in zip(x_values, channels_bpps)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"hyperprior bpps_per_channels" : wandb.plot.scatter(table, "x", "y",
                                 title="hyperprior bpps_per_channels")})
    
    
    

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def plot_different_values(model, test_dataloader, device ,dim = 128):


    # hyperprior
    cont = torch.zeros(model.N)
    ch = model.N

    print(cont.shape)
    
    unique_val_ch = []
    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            out_enc = model.g_a(d)
            out_enc = model.h_a(torch.abs(out_enc))
            bs, ch, w,h = out_enc.shape
            out_enc = out_enc.reshape(ch,bs,w*h)
            y_hype = model.entropy_bottleneck.sos(out_enc,-1)
            for j in range(ch):
                print(j)
                unique_val, number_val = torch.unique(y_hype[j,:,:],return_counts=True)                    
                cont[j] += unique_val.shape[0] 
                if j == dim:
                    unique_val_ch += unique_val

    
    cont = cont/len(test_dataloader)
    cont = cont.tolist() # convert to list 
    

    x_values = torch.arange(model.N).tolist()
    data = [[x, y] for (x, y) in zip(x_values, cont)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"hyperprior values per channel" : wandb.plot.scatter(table, "x", "y",
                                 title="hyperprior values per channel")})
    
    
    channel_val = torch.tensor(unique_val_ch)
    unique_val, number_val = torch.unique(channel_val,return_counts=True)
    number_val = number_val/torch.sum(number_val)
    x_values = unique_val.tolist()
    data = [[x, y] for (x, y) in zip(x_values,number_val.tolist())]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"hyperprior emp dix of channel " + str(dim) : wandb.plot.scatter(table, "x", "y",
                                 title= "hyperprior emp. dix. of channel " + str(dim))})  
    



def plot_difference_of_biases(model):
    # likelihood 
    b_values = model.entropy_bottleneck.sos.b
    w_values = model.entropy_bottleneck.sos.w
    res_b = torch.zeros(b_values.shape[0] - 1)
    res_w =torch.zeros(w_values.shape[0] - 1)
    for i in range(b_values.shape[0] - 1):
        res_b[i] = torch.abs(b_values[i + 1] - b_values[i])
        res_w[i] = torch.abs(w_values[i + 1] - w_values[i])
    
    x_values = torch.arange(b_values.shape[0] - 1)

    data = [[x, y] for (x, y) in zip(x_values,res_b)]
    table = wandb.Table(data=data, columns = ["x", "b_values"]) 
    wandb.log({"likelihood bias difference"  :wandb.plot.scatter(table, "x", "b_values", title="likelihood bias difference" )})

    data = [[x, y] for (x, y) in zip(x_values,res_w)]
    table = wandb.Table(data=data, columns = ["x", "w_values"]) 
    wandb.log({"likelihood weights difference":wandb.plot.scatter(table, "x", "w_values", title="likelihood weigths difference")})   



    b_values = model.gaussian_conditional.hyper_sos.b
    w_values = model.gaussian_conditional.hyper_sos.w
    res_b = torch.zeros(b_values.shape[0] - 1)
    res_w =torch.zeros(w_values.shape[0] - 1)
    for i in range(b_values.shape[0] - 1):
        res_b[i] = torch.abs(b_values[i + 1] - b_values[i])
        res_w[i] = torch.abs(w_values[i + 1] - w_values[i])
    
    x_values = torch.arange(b_values.shape[0] - 1)

    data = [[x, y] for (x, y) in zip(x_values,res_b)]
    table = wandb.Table(data=data, columns = ["x", "b_values"]) 
    wandb.log({"hyperprior bias difference"  :wandb.plot.scatter(table, "x", "b_values", title="hyperprior bias difference" )})

    data = [[x, y] for (x, y) in zip(x_values,res_w)]
    table = wandb.Table(data=data, columns = ["x", "w_values"]) 
    wandb.log({"hyperprior weights difference":wandb.plot.scatter(table, "x", "w_values", title="hyperprior weigths difference")})   
       



def plot_bias_and_weights(model):

    b_values = model.entropy_bottleneck.sos.b
    w_values = model.entropy_bottleneck.sos.w
    if model.entropy_bottleneck.sos.num_sigmoids == 0:
        x_values = torch.arange(model.entropy_bottleneck.sos.minimo, model.entropy_bottleneck.sos.massimo + 1)
    else:
        x_values = torch.arange(model.entropy_bottleneck.sos.num_sigmoids )
    

    data = [[x, y] for (x, y) in zip(x_values,b_values)]
    table = wandb.Table(data=data, columns = ["x", "b_values"]) 
    wandb.log({"actual bias"  :wandb.plot.scatter(table, "x", "b_values", title="actual bias" )})

    data = [[x, y] for (x, y) in zip(x_values,w_values)]
    table = wandb.Table(data=data, columns = ["x", "w_values"]) 
    wandb.log({"actual weights":wandb.plot.scatter(table, "x", "w_values", title="actual weights")})   




    b_values = model.gaussian_conditional.hyper_sos.b
    w_values = model.gaussian_conditional.hyper_sos.w
    if model.entropy_bottleneck.sos.num_sigmoids == 0:
        x_values = torch.arange(model.gaussian_conditional.hyper_sos.minimo,model.gaussian_conditional.hyper_sos.massimo + 1)
    else:
        x_values = torch.arange(model.gaussian_conditional.hyper_sos.num_sigmoids )
    

    data = [[x, y] for (x, y) in zip(x_values,b_values)]
    table = wandb.Table(data=data, columns = ["x", "b_values"]) 
    wandb.log({"hyperprior actual bias"  :wandb.plot.scatter(table, "x", "b_values", title="hyperprior actual bias" )})

    data = [[x, y] for (x, y) in zip(x_values,w_values)]
    table = wandb.Table(data=data, columns = ["x", "w_values"]) 
    wandb.log({"hyperprior actual weights":wandb.plot.scatter(table, "x", "w_values", title="hyperprior actual weights")})   
    




def plot_likelihood_sos(net, device,n = 1000, dim = 0):
    
    #likelihood latent space
    """
    minimo = - float(torch.sum(net.gaussian_conditional.hyper_sos.w).detach().cpu().numpy()) # not zero, but -massimo
    massimo = float(torch.sum(net.gaussian_conditional.hyper_sos.w).detach().cpu().numpy())
    space = (massimo-minimo)/n
    x_values = torch.arange(minimo, massimo, space)

    sample = x_values.repeat(net.M, 1).unsqueeze(1).to(device)

    y_values = net.gaussian_conditionalhyper_sos.d(sample)[dim, :].squeeze(0) #[1000]
    data = [[x, y] for (x, y) in zip(x_values,y_values)]
    table = wandb.Table(data=data, columns = ["x", "p_y"])
    wandb.log({ "likelihood at dimension " + str(dim): wandb.plot.line(table, "x", "p_y", title='likelihood at dimension ' + str(dim))})
    """
    # y latent space latent space
    minimo = - float(torch.sum(net.entropy_bottleneck.sos.w).detach().cpu().numpy()) # not zero, but -massimo
    massimo = float(torch.sum(net.entropy_bottleneck.sos.w).detach().cpu().numpy())
    space = (massimo-minimo)/n
    x_values = torch.arange(minimo, massimo, space)

    sample = x_values.repeat(net.N, 1).unsqueeze(1).to(device)

    y_values = net.entropy_bottleneck._likelihood(sample)[dim, :].squeeze(0) #[1000]
    data = [[x, y] for (x, y) in zip(x_values,y_values)]
    table = wandb.Table(data=data, columns = ["x", "p_y"])
    wandb.log({"hyperprior at dimension " + str(dim): wandb.plot.line(table, "x", "p_y", title='hyperprior at dimension ' + str(dim))})



      

