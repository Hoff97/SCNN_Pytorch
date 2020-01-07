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
from dataset.Phoenix import all_classes, lane_classes
from model import SCNN
from utils.lr_scheduler import PolyLR
from utils.transforms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp20")
    parser.add_argument("--model_path", type=str, default="./experiments/exp20/exp20.pth")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

model_path = args.model_path

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

# ------------ train data ------------
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)

dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
dataset = Dataset_Type(Dataset_Path[dataset_name], transform_train, **exp_cfg['dataset']['other'])

seg_classes = 5
if hasattr(dataset, 'seg_classes'):
    seg_classes = getattr(dataset, 'seg_classes')
net = SCNN(resize_shape, pretrained=True, seg_classes=seg_classes, weights=Dataset_Type.get_weights(exp_cfg['dataset']['other']['seg_mode']))

def main():
    print('Loading model')
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['net'])
    dummy_input = torch.zeros((2,3,resize_shape[0],resize_shape[1]))
    #net(dummy_input)
    torch.onnx.export(net, dummy_input, "model.onnx", opset_version=11)

if __name__ == "__main__":
    main()
