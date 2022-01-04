import torch
import matplotlib.pyplot as plt
import torchvision

from tqdm import tqdm
from models import UNET, small_ae, large_ae, cnn
from pl_bolts.models.autoencoders import AE
from piqa import SSIM

model_paths = ['model_runs/small_ae2_mse_1', 'model_runs/large_ae2_mse_1', 'model_runs/cnn2_MSE_1', 'model_runs/small_unet2_MSE_1', 'model_runs/resnet2_mse_1']
model_list = [small_ae(3 ,128, latent_dim=512), large_ae(3, 128, latent_dim = 512), cnn(), UNET(features=[16, 32, 64, 128]), AE(input_height=32)]
model_names = ['CNN/FNN Small', 'CNN/FNN Large', 'CNN', 'UNET', 'Resnet18']

def plot_results(data_loader, model_list = model_list, model_paths = model_paths, model_names= model_names, normalize= False, pixel_range=(0,1)):
    
    # get some random training images
    noisy, _ , input = next(iter(data_loader))
    recons = []
    for i, (model, name, path) in enumerate(zip(model_list, model_names, model_paths)):
        PATH = path+'.pth'
        model.load_state_dict(torch.load(PATH))

        model.eval()
        
        with torch.no_grad():
            recon = model(noisy)
        recons.append(recon)


    # Plotting
    imgs = torch.stack([input, noisy, recons[0], recons[1]], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=normalize, value_range=pixel_range)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12,8))
    #plt.title(f'{title}\ninput/noisy/cleaned', fontsize = 18)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
    

MSE = torch.nn.MSELoss(reduction='mean')
ssim = SSIM()

def plot_recon(model, data_loader, title = 'test', normalize = False, pixel_range = (0,1)):
    
    # get some random training images
    noisy, _ , input = next(iter(data_loader))

    # Reconstruct images
    model.eval()
    
    with torch.no_grad():
        recon = model(noisy)
    recon = recon

    print(f'SSIM: {ssim(recon, input).numpy()}')
    print(f'MSE: {MSE(recon, input).numpy()}')

    # Plotting
    imgs = torch.stack([input, noisy, recon], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=normalize, value_range=pixel_range)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(10,6))
    plt.title(f'{title}\ninput/noisy/cleaned', fontsize = 18)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
    