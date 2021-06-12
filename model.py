import segmentation_models_pytorch as smp
import torch.nn as nn

# class SegNet(nn.Module):
# model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

# import cv2


class SegNet(nn.Module):
    def __init__(self, model_name="resnet18", classes=23, pretrained=True):
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
        # bs = x.size(0)
        # features = self.model(x)
        # pooled_features = self.pooling(features).view(bs, -1)
        # output = self.fc(pooled_features)
        return self.model(x)
