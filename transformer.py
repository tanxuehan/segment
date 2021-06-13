import albumentations as A
import cv2

from config import *

t_train = A.Compose(
    [
        A.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        A.GaussNoise(),
    ]
)

t_val = A.Compose(
    [
        A.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.GridDistortion(p=0.2),
    ]
)
