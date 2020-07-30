import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
import torchvision.models.vgg as vgg
import os


def conv5x5(in_channels, out_channels, mode=None, sigmoid=False):
    ops = [nn.Conv2d(in_channels, out_channels, 5, padding=2),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()]
    if mode == "down":
        ops.insert(0, nn.MaxPool2d(2))
    elif mode == "up":
        ops.insert(0, nn.Upsample(scale_factor=2))
    if sigmoid:
        ops.pop(-1)
        ops.append(nn.Tanh())
    return nn.Sequential(*ops)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip=False):
        super(Decoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 128, "up")
        self.conv3 = conv5x5(128, 64, "up")
        self.conv5 = conv5x5(64, 32, "up")
        self.conv7 = conv5x5(32, out_channels, "up", sigmoid=True)
        if skip:
            self.conv2 = conv5x5(256, 128)
            self.conv4 = conv5x5(128, 64)
            self.conv6 = conv5x5(64, 32)
        else:
            self.conv2 = conv5x5(128, 128)
            self.conv4 = conv5x5(64, 64)
            self.conv6 = conv5x5(32, 32)
        self.skip = skip

    def forward(self, input, skip):
        if self.skip:
            assert skip is not None and len(skip) == 3
            output = self.conv2(torch.cat([skip[0], self.conv1(input)], 1))
            output = self.conv4(torch.cat([skip[1], self.conv3(output)], 1))
            output = self.conv6(torch.cat([skip[2], self.conv5(output)], 1))
            return self.conv7(output)
        else:
            return self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(input)))))))

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip=False):
        super(Encoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

        self.skip = skip

    def forward(self, input):
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(output[-1]))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(output[-1]))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(output[-1]))
        output.append(self.conv8(output[-1]))
        if self.skip:
            return output[-1], [output[5], output[3], output[0]]
        return output[-1], None


class Pose_to_Image(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, skip=False):
        super(Pose_to_Image, self). __init__()
        self.encoder_conv = Encoder(pose_channels+img_channels, 256, skip)
        self.decoder_conv = Decoder(256, 3, skip)
        self.skip = skip

    def forward(self, input):
        pose_imgs, human_imgs, background_imgs, masks = input
        output, skip = self.encoder_conv(torch.cat([pose_imgs, human_imgs+background_imgs], 1))
        output = self.decoder_conv(output, skip)
        return [output]

class Perceptual(nn.Module):
    def __init__(self, mapping):
        super(Perceptual, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layers_mapping = mapping

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layers_mapping:
                output[self.layers_mapping[name]] = x
        return output

