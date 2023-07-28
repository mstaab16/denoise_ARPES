import torch
import torch.nn as nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

import matplotlib.pyplot as plt


from sim_ARPES import generate_batch


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSSIM = MultiScaleStructuralSimilarityIndexMeasure()
        self.MSE = nn.MSELoss()
        self.alpha = 0.3

    def forward(self, pred, target):
        return self.alpha * self.MSE(pred, target) + (1 - self.alpha) * self.MSSIM(pred, target)

lossFunc = CombinedLoss()

X, Y = generate_batch(1, noise=0.1, k_resolution=0.01, e_resolution=0.005)
X = torch.unsqueeze(torch.tensor(X),1)
Y = torch.unsqueeze(torch.tensor(Y),1)
loss = lossFunc(X, Y)
# loss = nn.MSELoss()(X[0], Y[0])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(X[0,0], cmap='gray_r', origin='lower')
ax1.set_title(f'Input min: {X[0,0].min():.2f}, max: {X[0,0].max():.2f}')
ax2.imshow(Y[0,0], cmap='gray_r', origin='lower')
ax2.set_title(f'Target min: {Y[0,0].min():.2f}, max: {Y[0,0].max():.2f}')
ax3.imshow((X[0,0]-Y[0,0])**2, cmap='gray_r', origin='lower')
ax3.set_title(f'Inferred min: {((X[0,0]-Y[0,0])**2).min():.2f}, max: {((X[0,0]-Y[0,0])**2).max():.2f}')
fig.suptitle(f'Loss: {loss:.2f}')
plt.show()
