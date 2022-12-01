import torch 

import numpy as np


delta = 0.50
delta_m = delta/2
x = torch.tensor([ 2.0100,  2.5600,2.4900,  0.0040,  0.6200, -1.6700, -1.0400, -1.2200])
x = x.repeat(128).reshape(128,1,8)
b = torch.arange(-20,30, delta)


#d = torch.stack([(delta*torch.relu(1 - (2/delta)*torch.abs(x - b[i]))) - delta_m for i in range(b.shape[0])], dim=0).sum(dim=0)
#print(b)
#print(x)
f = torch.zeros(x.shape)
#for i in range(b.shape[0]):
#    f = f + torch.sign(torch.relu(1 - (2/delta)*torch.abs(x - b[i])))*b[i]

    #print(torch.sign(torch.relu(1 - (2/delta)*torch.abs(x - b[i])))*b[i])



f = torch.sum(torch.sign(torch.relu(1 - (2/delta)*torch.abs(x - b[None,:,None])))*b[None,:,None], dim = 1).unsqueeze(1)
print(f.shape) 
print(torch.unique(f))
print(x[0,0,:])
print(f[0,0,:])
#print(x)   
#print(torch.sign(torch.relu(1 - (2/delta)*torch.abs(x - b[1])))*b[1])


x = torch.arange(-5,5,0.001)
lev = torch.arange(-6,6)
p = 1 
h = 1 

def f(x,l, p = 1, h = 1):
    res = torch.zeros(x.shape[0])
    for l in lev:
        res += torch.relu(1 - torch.pow((2/h)*torch.abs(x - l),p))
    return res 


import matplotlib.pyplot as plt
res = f(x,lev)
fig, ax = plt.subplots(figsize=(18, 6))
plt.plot(x, res)
plt.savefig("/Users/albertopresta/Desktop/icme/prova.png")


    