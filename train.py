import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchgeometry.losses.dice import DiceLoss
from tqdm import tqdm

from config import *
from dataset import SegDataset
from dice_coefficient import dice_coefficient
from model import SegNet
from transformer import *


def train_func(train_loader):
    """
    Train for one epoch.
    """
    model.train()
    bar = tqdm(train_loader)
    if use_amp:
        torch.cuda.amp.GradScaler()
    losses = []
    for _, (images, targets) in enumerate(bar):

        images, targets = images.to(device), targets.to(device)

        predict = model(images)
        loss = criterion(predict, targets)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f"loss: {loss.item():.5f}, smth: {smooth_loss:.5f}")

    loss_train = np.mean(losses)
    return loss_train


def valid_func(valid_loader):
    """
    Validation
    """
    model.eval()
    bar = tqdm(valid_loader)
    losses = []
    coeffs = []
    with torch.no_grad():
        for images, targets in bar:
            images, targets = images.to(device), targets.to(device)
            predict = model(images)
            coeffs.append(dice_coefficient(predict, targets))

            loss = criterion(predict, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f"loss: {loss.item():.5f}, smth: {smooth_loss:.5f}")

    loss_valid = np.mean(losses)
    coeff_valid = np.mean(coeffs)
    return loss_valid, coeff_valid


def start_train():
    names = []
    for dirname, _, filenames in os.walk(img_path):
        for filename in filenames:
            names.append(filename.split(".")[0])

    X_train, X_val = train_test_split(names, test_size=0.1, random_state=19)

    train_set = SegDataset(X_train, transform=t_train)
    val_set = SegDataset(X_val, transform=t_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=valid_batch_size, shuffle=True)

    log = {}
    loss_min = 99999
    not_improving = 0
    coeff_max = 0.0

    for epoch in range(1, n_epochs + 1):
        loss_train = train_func(train_loader)
        loss_valid, coeff = valid_func(val_loader)

        log["loss_train"] = log.get("loss_train", []) + [loss_train]
        log["loss_valid"] = log.get("loss_valid", []) + [loss_valid]
        log["lr"] = log.get("lr", []) + [optimizer.param_groups[0]["lr"]]
        log["coeff"] = log.get("coeff", []) + [coeff]
        content = (
            time.ctime()
            + " "
            + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, coeff: {coeff:.6f}.'
        )
        print(content)
        not_improving += 1

        if coeff > coeff_max:
            print(f"coeff_max ({coeff_max:.6f} --> {coeff:.6f}). Saving model ...")
            torch.save(model.state_dict(), f"{model_dir}_best_coeff_{coeff:.6f}.pth")
            coeff_max = coeff
            not_improving = 0

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(model.state_dict(), f"{model_dir}_best_loss.pth")

        if not_improving == early_stop:
            print("Early Stopping...")
            break

    torch.save(model.state_dict(), f"{model_dir}_final_.pth")


if __name__ == "__main__":
    model = SegNet()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    start_train()
