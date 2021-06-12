import os

import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import *
from dataset import SegDataset
from model import SegNet

cudnn.benchmark = True

ckpt_path = "./weights/"
eval_path = "./eval_res/"

args = {"snapshot": "_best_loss.pth"}


def eval():
    model = SegNet()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("load model " + args["snapshot"])
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args["snapshot"])))
    model.eval()

    X_test = []
    for dirname, _, filenames in os.walk(test_path):
        for filename in filenames:
            X_test.append(filename.split(".")[0])

    test_set = SegDataset(X_test, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    with torch.no_grad():
        for vi, img in enumerate(test_loader):
            img = img.to(device)
            output = model(img)
            prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            print(prediction)


if __name__ == "__main__":
    eval()
