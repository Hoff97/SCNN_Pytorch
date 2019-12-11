import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from config import *
from dataset.Phoenix import all_classes
from model import SCNN
from utils.lr_scheduler import PolyLR
from utils.tensorboard import TensorBoard
from utils.transforms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], transform_train, visualize=True, **exp_cfg['dataset']['other'])

# ------------ preparation ------------
seg_classes = 5
if hasattr(train_dataset, 'seg_classes'):
    seg_classes = getattr(train_dataset, 'seg_classes')

def main():
    for i in range(1):
        t = train_dataset[i]
        print(i)


if __name__ == "__main__":
    main()
