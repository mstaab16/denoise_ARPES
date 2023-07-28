import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import time
import matplotlib.pyplot as plt
import wandb

from unet import UNet
from dataset import FakeARPESDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 2
initial_learning_rate = 0.001

noise=0.1
k_resolution=0.01
e_resolution=0.005

wandb.init(
    # set the wandb project where this run will be logged
    project="ARPES-denoise",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "UNet",
    "dataset": "random ARPES",
    "epochs": num_epochs,
    "batch_size": batch_size,
    "noise": noise,
    "k_resolution": k_resolution,
    "e_resolution": e_resolution,
    }
)



# initialize our UNet model
unet = UNet().to(device)
# initialize loss function and optimizer
# lossFunc = MultiScaleStructuralSimilarityIndexMeasure().to(device)
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSSIM = MultiScaleStructuralSimilarityIndexMeasure()
        self.MSE = nn.MSELoss()
        self.alpha = 0.3

    def forward(self, pred, target):
        return self.alpha * self.MSE(pred, target) + (1 - self.alpha) * self.MSSIM(pred, target)

lossFunc = CombinedLoss().to(device)

opt = Adam(unet.parameters(), lr=initial_learning_rate)
# calculate steps per epoch for training and test set

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# create the training and test data loaders

bestTestLoss = 1e10

train_dataset = FakeARPESDataset(n=25000, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution)
test_dataset = FakeARPESDataset(n=5000, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution)
trainSteps = len(train_dataset) // batch_size
testSteps = len(test_dataset) // batch_size
trainLoader = DataLoader(train_dataset, shuffle=False,
    batch_size=batch_size, pin_memory=True,
    num_workers=0)
testLoader = DataLoader(test_dataset, shuffle=False,
    batch_size=batch_size, pin_memory=True,
    num_workers=0)

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in range(num_epochs):
    # if e % 10 == 0:
    #     print("[INFO] generating new dataset...")
    #     train_dataset = FakeARPESDataset(n=1600, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution)
    #     test_dataset = FakeARPESDataset(n=400, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution)
    #     trainSteps = len(train_dataset) // batch_size
    #     testSteps = len(test_dataset) // batch_size
    #     trainLoader = DataLoader(train_dataset, shuffle=False,
    #         batch_size=batch_size, pin_memory=False,
    #         num_workers=0)
    #     testLoader = DataLoader(test_dataset, shuffle=False,
    #         batch_size=batch_size, pin_memory=False,
    #         num_workers=0)
    print("[INFO] starting epoch {}/{}...".format(e, num_epochs))
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        print("[INFO] starting epoch {}/{}... \t {}/{} \t ellapsed:{} \t X.shape: {} ".format(e, num_epochs,i,trainSteps,time.time()-startTime, x.shape))
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        wandb.log({"train_loss": loss})
        totalTrainLoss += loss
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in testLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            loss = lossFunc(pred, y)
            totalTestLoss += loss
            wandb.log({"test_loss": loss})
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss))
    if avgTestLoss < bestTestLoss:
        bestTestLoss = avgTestLoss
        torch.save(unet.state_dict(), "unet.pth")
    wandb.log({"average_train_loss": avgTrainLoss, "average_test_loss": avgTestLoss})
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
wandb.finish()

plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()