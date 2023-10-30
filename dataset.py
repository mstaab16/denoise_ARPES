from torch.utils.data import Dataset
import torch
from sim_ARPES import generate_batch
import numpy as np

class FakeARPESDataset(Dataset):
    def __init__(self, n, noise=None, k_resolution=None, e_resolution=None):
        self.n = n
        self.spectra, self.targets = generate_batch(n, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution, num_angles=1024, num_energies=1024)
        self.spectra = torch.unsqueeze(torch.from_numpy(self.spectra), 1)
        self.targets = torch.unsqueeze(torch.from_numpy(self.targets), 1)

    def __len__(self):
        # return the number of total samples contained in the dataset
        return self.n
    
    def __getitem__(self, idx):
        
        spectrum = self.spectra[idx]
        target = self.targets[idx]
        return (spectrum, target)


from qlty import qlty2D

class NpyDataset(Dataset):
    def __init__(self, start_idx, end_idx, Y=1024, X=1024, window=(256,256), step=(128,128), border=None, border_weight=0):
        self.qlty_obj = qlty2D.NCYXQuilt(Y=Y, X=X, window=window, step=step, border=border, border_weight=border_weight)
        self.num_indices_per_file = int((X/step[0])*(Y/step[1]))

        self.x_files = [f'data\\x_{i:04d}.npy' for i in range(start_idx, end_idx)]
        self.y_files = [f'data\\y_{i:04d}.npy' for i in range(start_idx, end_idx)]


        self.most_recent_filenum = None
    
    def __getitem__(self, index):
        filenum = index//self.num_indices_per_file
        # print(self.x_files[filenum], filenum, index)

        if filenum == self.most_recent_filenum:
            return self.qx[index % self.num_indices_per_file], self.qy[index % self.num_indices_per_file]

        self.most_recent_filenum = filenum
        x = np.load(self.x_files[filenum])
        x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x).float(),0),0)
        y = np.load(self.y_files[filenum])
        y = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(y).float(),0),0)
        self.qx, self.qy = self.qlty_obj.unstitch_data_pair(x,y) 
        return self.qx[index % self.num_indices_per_file]* self.num_indices_per_file, self.qy[index % self.num_indices_per_file]* self.num_indices_per_file

    def __len__(self):
        return len(self.x_files * self.num_indices_per_file)
    

if __name__ == '__main__':
    d = NpyDataset(0,10)
    from time import perf_counter_ns
