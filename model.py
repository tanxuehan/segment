import segmentation_models_pytorch as smp
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, model_name="vgg16_bn", classes=23, pretrained=True):
        super().__init__()
        self.model = smp.Unet(
            model_name,
            encoder_weights="imagenet",
            classes=classes,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )

    def forward(self, x):
        return self.model(x)
