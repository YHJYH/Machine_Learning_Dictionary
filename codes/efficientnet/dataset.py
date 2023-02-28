import os
import glob
import numpy as np
from PIL import Image
from typing import Any, Optional, Callable, Tuple

import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256

class Caltech256Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str = None, download=True, transform: Optional[Callable] = None) -> None:
        
        self.root = root
        self.transform = transform
        subfolder = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        # self.dataset = Caltech256(root=root, download=download, transform=transform)
        
        self.data: Any = []
        self.label = [] 
        
        for folder in subfolder:
            jpg_files = [f for f in os.listdir(root + '\\' + folder)]
            for file in jpg_files:
                img = Image.open(root + '\\' + folder + '\\' + file)
                img = img.convert('RGB')
                self.data.append(np.array(img))
                self.label.append(int(file[:3]))
                img.close()
        
        #for i in range(len(self.dataset)):
         #   self.data.append(self.dataset[i][0]) # img, type=numpy.ndarray, shape=(H, W, C)
          #  self.label.append(self.dataset[i][1]) # label, type=int

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        # img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.data)
    
