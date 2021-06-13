import numpy as np
import torch
import torch.nn.functional as F


def dice_coefficient(pred_mask, mask, smooth=1e-10, n_classes=23):
    """
    计算dice coefficient。
    Parameters:
        pred_mask: 预测值
        mask: gt
        smooth: eps
        n_classes: 类别数
    Returns:
        float: dice coefficient
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)

        coefficients = []
        for cls in range(0, n_classes):
            true_class = pred_mask == cls
            true_label = mask == cls

            if true_label.long().sum().item() == 0:
                continue
            else:
                intersect = (
                    torch.logical_and(true_class, true_label).sum().float().item()
                )
                union = torch.logical_or(true_class, true_label).sum().float().item()

                coefficient = (intersect + smooth) / (union + smooth)
                coefficients.append(coefficient)

        return np.mean(coefficients)
