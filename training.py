import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optimizer
from models import UNET
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from piqa import SSIM

mse_fn = nn.MSELoss()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

model = UNET()
model.to(device)

lr= 1e-4
w_decay = 1e-5
optim = optimizer.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
scheduler = optimizer.lr_scheduler.ExponentialLR(optim, gamma = 0.9, last_epoch=- 1, verbose=False)

def train(model, device, dataloader, optimizer, loss_1 = mse_fn, loss_2 = 0):

    model.train()
    train_loss = []
    # loop over data loader
    for x_tran, _, x_clean in dataloader:

        # move transformed image to device
        x_tran = x_tran.to(device)
        x_clean = x_clean.to(device)
        
        # obtain reconstruction from model
        x_recon = model(x_tran)

        # combined loss

        ssim = 1 - loss_2(x_recon, x_clean)
        mse = loss_1(x_recon, x_clean)
        loss = mse + ssim
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

def test(model, device, dataloader, loss_1 = mse_fn, loss_2 = 0, plot = False):

    model.eval()

    with torch.no_grad():
        
        x_recon_list, x_clean_list, x_tran_list = [], [], []
        val_loss = []
        for x_tran, _, x_clean in dataloader:
            
            # move transformed image to device
            x_tran = x_tran.to(device)
            x_clean = x_clean.to(device)

            # obtain reconstruction from model
            x_recon = model(x_tran)

            # combined loss
            ssim = 1 - loss_2(x_recon, x_clean)
            mse = loss_1(x_recon, x_clean)
            loss = mse + ssim

            # Append the network output and the original image to the lists
            x_recon_list.append(x_recon.cpu())
            x_clean_list.append(x_clean.cpu())
            x_tran_list.append(x_tran.cpu())
      
        # Evaluate global loss
        val_loss.append(loss.detach().cpu().numpy())

    scheduler.step()

    x_recon_list = torch.cat(x_recon_list)
    x_clean_list = torch.cat(x_clean_list) 
    x_tran_list = torch.cat(x_tran_list)
     
    # Plotting
    if plot == True:
        imgs = torch.stack([x_clean_list[:6], x_tran_list[:6], x_recon_list[:6]], dim=1).flatten(0,1)
        grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=False, value_range=(0,1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(12,6))
        #plt.title(f'{str(model)}\ninput/noisy/cleaned', fontsize = 12)
        plt.imshow(grid)
        plt.axis('off')
        plt.show()

    return np.mean(val_loss)