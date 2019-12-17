import argparse
import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = image[:,0:3]
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

    def set_target(self, target):
        self.target = target.detach()

def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv3']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    #optimizer = optim.LBFGS([input_img.requires_grad_()])
    optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        #optimizer.step(closure)
        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", type=str, default="./res/phoenix/top/rgb")
    parser.add_argument("--style", "-s", type=str, default="./res/style/frame0296.png")
    parser.add_argument("--output", "-o", type=str, default="./res/phoenix/top/rgb_style")
    parser.add_argument("--vis", "-v", action='store_true')
    parser.add_argument("--number", "-n", type=int, default=-1)
    parser.add_argument("--log", "-l", action='store_true')
    args = parser.parse_args()

    vis = args.vis
    style = args.style
    imgs_dir = args.imgs
    n = args.number
    out_dir = args.output
    log = args.log


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    style_imgs = None
    if os.path.isfile(style):
        style_imgs = [image_loader(style)]
    elif os.path.isdir(style):
        style_imgs = [image_loader(os.path.join(style, name)) for name in os.listdir(style)]

    content_imgs_name = os.listdir(imgs_dir)
    content_imgs_name.sort()
    content_imgs = [image_loader(os.path.join(imgs_dir, name)) for name in content_imgs_name]

    assert style_imgs[0].size() == content_imgs[0].size(), \
        "we need to import style and content images of the same size"


    unloader = transforms.ToPILImage()  # reconvert into PIL image

    if vis:
        plt.figure()
        imshow(style_imgs[0], title='Style Image')

    if vis:
        plt.figure()
        imshow(content_imgs[0], title='Content Image')

    style_imgs = torch.stack(style_imgs, dim=1).squeeze()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    for i, content_img in enumerate(content_imgs[:n]):
        print(f'Stylizing {i+1}/{len(content_imgs[:n])}')

        input_img = content_img.clone() + torch.randn(content_img.data.size(), device=device)*0.02
        input_img = torch.min(torch.max(input_img, torch.zeros_like(input_img)), torch.ones_like(input_img))

        if vis:
            plt.figure()
            imshow(input_img, title='Input Image')


        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                       content_img, style_imgs, input_img)

        if vis:
            plt.figure()
            imshow(output, title='Output Image')
            plt.show()

        image = output.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        image.save(os.path.join(out_dir, content_imgs_name[i]))

if __name__ == "__main__":
    main()
