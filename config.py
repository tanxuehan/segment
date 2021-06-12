import torch

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_height = 704
img_width = 1056
img_path = "/data/segment_q/data/train/images/"
mask_path = "/data/segment_q/data/train/masks/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_lr = 1e-3
n_epochs = 15
weight_decay = 1e-4
use_amp = False
batch_size = 3
valid_batch_size = 3
early_stop = 5
model_dir = "weights/"
test_path = "/data/segment_q/data/test/images/"
