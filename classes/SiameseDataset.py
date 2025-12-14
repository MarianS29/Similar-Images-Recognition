import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img1_path = self.df.loc[idx, "path"]

        # 1 = pereche pozitivă (aceeași imagine), 0 = negativă (imagine diferită)
        same_image = random.randint(0, 1)

        if same_image == 1:
            img2_path = img1_path
            target = 1.0
        else:
            idx2 = idx
            while idx2 == idx:
                idx2 = random.randint(0, len(self.df) - 1)
            img2_path = self.df.loc[idx2, "path"]
            target = 0.0

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        target = torch.tensor([target], dtype=torch.float32)
        return img1, img2, target
