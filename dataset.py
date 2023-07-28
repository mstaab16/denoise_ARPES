from torch.utils.data import Dataset
import torch
from sim_ARPES import generate_batch

class FakeARPESDataset(Dataset):
	def __init__(self, n, noise=None, k_resolution=None, e_resolution=None):
		self.n = n
		self.spectra, self.targets = generate_batch(n, noise=noise, k_resolution=k_resolution, e_resolution=e_resolution)
		self.spectra = torch.unsqueeze(torch.from_numpy(self.spectra), 1)
		self.targets = torch.unsqueeze(torch.from_numpy(self.targets), 1)

	def __len__(self):
		# return the number of total samples contained in the dataset
		return self.n
	
	def __getitem__(self, idx):
		
		spectrum = self.spectra[idx]
		target = self.targets[idx]
		return (spectrum, target)