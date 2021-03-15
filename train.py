#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 14:01:52 2021

@author: nuvilabs
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from loss import ContentLoss, StyleLoss
from utils import get_style_model_and_losses
import argparse
parser = argparse.ArgumentParser(description='Pytorch style transfer')
parser.add_argument('--style', default="style.jpg", type=str, help='style img')
parser.add_argument('--content', default="content.jpg", type=str, help='content img')
parser.add_argument('--epochs', default=600, type=int, help='epochs')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader('images/' + args.style)
content_img = image_loader('images/' + args.content)
print(style_img.size(),content_img.size())
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    if title is not None:
        import cv2
        import numpy as np
        image = np.asarray(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite('results/' + title+'.png',image)
    else:
        plt.title(title)
        plt.imshow(image)


cnn = models.vgg19(pretrained=True).features.to(device).eval()


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
input_img = torch.randn(content_img.data.size(), device=device)
input_img = content_img.clone()
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer



def train(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=args.epochs,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    epoch = [0]
    images = []
    while epoch[0] <= num_steps:

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

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("epoch {}:".format(epoch))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            if epoch[0] % 10 == 0 or epoch[0] < 10:
                img = input_img.cpu().clone().clamp_(0, 1)
                img = img.squeeze(0)      # remove the fake batch dimension
                img = unloader(img)
                images.append(img)
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, images

output, images = train(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

name = args.content.split('.')[0] + '_' + args.style.split('.')[0]
imshow(output, title=name)

images[0].save(f'results/{name}.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)