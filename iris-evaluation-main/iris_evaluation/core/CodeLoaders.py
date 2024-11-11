"""
Written by Rob Hart of Walsh Lab @ IU Indianapolis.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class CodeLoader(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = list(iris_codes.files)
        self.names.sort()
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] #real + imaginary portions
        self.width = iris_codes[self.names[0]].shape[3]        

    def __len__(self):
        return len(self.names)


class RolledCodes(CodeLoader):
    def __getitem__(self, idx):
        iris_code = torch.empty((self.width, self.height * self.width), dtype=torch.bool) #allocates memory
        for i in range(self.width):
            iris_code[i] = torch.from_numpy(np.roll(self.iris_codes[self.names[idx]], shift=i, axis=3).reshape(1, self.height*self.width)) #roll, flatten, store

        return iris_code, self.names[idx]
    
class UnrolledCodes(CodeLoader):
    def __getitem__(self, idx):
        iris_code = torch.from_numpy(self.iris_codes[self.names[idx]].reshape(-1))

        return iris_code, self.names[idx]