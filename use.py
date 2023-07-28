import torch
import matplotlib.pyplot as plt

from sim_ARPES import generate_batch
from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, Y = generate_batch(1, noise=0.1, k_resolution=0.01, e_resolution=0.005)
X = torch.unsqueeze(torch.tensor(X).to(device),1)

def main(model):

    with torch.no_grad():
        Y_inferred = model(X)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(X[0, 0, :, :].cpu().numpy(), cmap='gray_r', origin='lower')
    ax1.set_title(f'Input min: {X[0, 0, :, :].min():.2f}, max: {X[0, 0, :, :].max():.2f}')
    ax2.imshow(Y[0], cmap='gray_r', origin='lower')
    ax2.set_title(f'Target min: {Y[0].min():.2f}, max: {Y[0].max():.2f}')
    ax3.imshow(Y_inferred[0, 0, :, :].cpu().numpy(), cmap='gray_r', origin='lower')
    ax3.set_title(f'Inferred min: {Y_inferred[0, 0, :, :].min():.2f}, max: {Y_inferred[0, 0, :, :].max():.2f}')
    plt.show()

if __name__ == "__main__":
    model = UNet()
    model.load_state_dict(torch.load('unet.pth'))
    model.to(device).eval()
    main(model)
