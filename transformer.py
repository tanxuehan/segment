
import albumentations
import albumentations as A
from albumentations import *
from config import *

transforms_train = albumentations.Compose([
   albumentations.RandomResizedCrop(img_height, img_width, scale=(0.9, 1), p=1), 
   albumentations.HorizontalFlip(p=0.5),
   albumentations.ShiftScaleRotate(p=0.5),
   albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
   albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
   albumentations.CLAHE(clip_limit=(1,4), p=0.5),
   albumentations.OneOf([
       albumentations.OpticalDistortion(distort_limit=1.0),
       albumentations.GridDistortion(num_steps=5, distort_limit=1.),
       albumentations.ElasticTransform(alpha=3),
   ], p=0.2),
   albumentations.OneOf([
       albumentations.GaussNoise(var_limit=[10, 50]),
       albumentations.GaussianBlur(),
       albumentations.MotionBlur(),
       albumentations.MedianBlur(),
   ], p=0.2),
  albumentations.Resize(img_height, img_width),
  albumentations.OneOf([
      JpegCompression(),
      Downscale(scale_min=0.1, scale_max=0.15),
  ], p=0.2),
  IAAPiecewiseAffine(p=0.2),
  IAASharpen(p=0.2),
  albumentations.Cutout(max_h_size=int(img_height * 0.1), max_w_size=int(img_width * 0.1), num_holes=5, p=0.5),
  albumentations.Normalize(),
])

transforms_valid = albumentations.Compose([
    albumentations.Resize(img_height, img_width),
    albumentations.Normalize()
])


t_train = A.Compose([A.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])

t_val = A.Compose([A.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])
