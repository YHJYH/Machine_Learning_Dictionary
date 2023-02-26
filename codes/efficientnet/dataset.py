import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256

class Caltech256Dataset(torch.utils.data.Dataset):
    def __init__(self, root=None, download=True, transform=None):
        self.root = root
        self.download = download
        self.transform = transform
        self.dataset = Caltech256(root=root, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)