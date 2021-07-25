import os
import re
from torch.utils.data import Dataset
from PIL import Image


class CatDog(Dataset):
    def __init__(self, root, transform=None):
        self.images = os.listdir(root)
        self.images.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        img = Image.open(os.path.join(self.root, file))

        if "dog" in file:
            label = 1
        elif "cat" in file:
            label = 0
        else:
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label
