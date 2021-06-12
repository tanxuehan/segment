import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import *


class SegDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def norm(self, img):
        """
        使用ImageNet的均值和标准差做归一化。
        """

        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return transform(img)

    def __getitem__(self, index):
        img = cv2.imread(img_path + self.X[index] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path + self.X[index] + ".png", cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        img = self.norm(img)
        mask = torch.from_numpy(mask).long()

        return img, mask
