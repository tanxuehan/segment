import os

import cv2
import torch.nn as nn
from fire import Fire
from torch.utils.data import DataLoader

from config import *
from dataset import SegDataset
from model import SegNet

args = {"snapshot": "_best_loss.pth"}


def eval(model_path="./weights/_best_loss.pth", eval_res_path="./eval_res/"):
    """
    evaluation
    命令行工具：python3 -m segment.eval model_path eval_res_path
    """
    model = SegNet()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("load model " + model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model_name = model_path.split("/")[-1]
    model_name = model_name.split(".")[0]

    X_test = []
    for dirname, _, filenames in os.walk(test_path):
        for filename in filenames:
            X_test.append(filename.split(".")[0])

    test_set = SegDataset(X_test, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    with torch.no_grad():
        for i, img in enumerate(test_loader):
            img = img.to(device)
            output = model(img)
            prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            os.makedirs(eval_res_path + model_name, exist_ok=True)
            save_path = eval_res_path + model_name + "/" + X_test[i] + ".png"
            print(save_path)

            cv2.imwrite(save_path, prediction)


if __name__ == "__main__":
    Fire(eval)
