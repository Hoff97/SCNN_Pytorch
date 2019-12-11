import argparse

import cv2
import torch

from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *
from dataset.Phoenix import Phoenix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    parser.add_argument("--mode", '-m', type=str, help="Segmentation mode (default or lanes)", default="default")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path
    mode = args.mode

    colors = Phoenix.get_colors(mode)
    colors = np.array(colors)
    num_classes = colors.shape[0]
    print(num_classes)

    net = SCNN(input_size=(512, 384), pretrained=False, seg_classes=num_classes, weights=Phoenix.get_weights(mode))

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    transform_img = Compose(Resize((512, 384)), Rotation(2))
    transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_img({'img': img})['img']
    x = transform_to_net({'img': img})['img']
    x.unsqueeze_(0)

    save_dict = torch.load(weight_path, map_location='cpu')

    net.load_state_dict(save_dict['net'])
    net.eval()

    seg_pred, exist_pred = net(x)[:2]
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]
    exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(num_classes)]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)

    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, num_classes):
        #if exist_pred[0, i] > 0.5:
        lane_img[coord_mask == (i + 1)] = colors[i]
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    cv2.imwrite("demo/demo_result.jpg", img)

    """for x in getLane.prob2lines_CULane(seg_pred, exist):
        print(x)"""

    if args.visualize:
        print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(num_classes)])
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
