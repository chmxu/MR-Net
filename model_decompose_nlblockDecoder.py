import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import torchvision.models.vgg as vgg
import torch.optim as optim
from resnet import BasicBlock
import os
from torchvision.ops import roi_pool
import sys
from model_DVG import Pose_to_Image as Pose_to_Image_DVG
from model_decompose_dilate import Pose_to_Image as Pose_to_Image_SinglePath

def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):    # pytorch version use for loop !!!
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    rois[:, 1:] = rois[:, 1:].clone().mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1).clone()[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)
    return output


def conv5x5(in_channels, out_channels, mode=None, sigmoid=False):
    ops = [nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)]
    if mode == "down":
        ops.insert(0, nn.MaxPool2d(2))
    elif mode == "up":
        ops.insert(0, nn.Upsample(scale_factor=2))
    if sigmoid:
        ops.pop(-1)
        ops.append(nn.Tanh())
    return nn.Sequential(*ops)

def conv3x3(in_channels, out_channels, stride=1, rate=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=rate),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True))

def conv1x1(in_channels, out_channels, full_seq=True, zero_init=False):
    if full_seq:
        ops = [nn.Conv2d(in_channels, out_channels, 1),
                            nn.BatchNorm2d(out_channels),
                                nn.ReLU(True)]
        if zero_init:
            nn.init.constant_(ops[1].weight, 0.)
        return nn.Sequential(*ops)
    else:
        return nn.Conv2d(in_channels, out_channels, 1)

class NonlocalBlock(nn.Module):
    def __init__(self, in_channels, scale_factor, different_src=False):
        super(NonlocalBlock, self).__init__()
        self.src_attn_conv = conv1x1(in_channels, in_channels//scale_factor, full_seq=False)    #theta
        self.guide_attn_conv = conv1x1(in_channels, in_channels//scale_factor, full_seq=False)  #phi
        self.src_conv = conv1x1(in_channels, in_channels//scale_factor, full_seq=False)             #g
        self.output_conv = conv1x1(in_channels//scale_factor, in_channels, zero_init=True)
        self.different_src = different_src
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        if self.different_src:
            src, guide = input
            batch_size, num_channel, w, h = src.shape
            src_attn = self.src_attn_conv(src).view(batch_size, -1, w*h).permute(0, 2, 1)
            guide_attn = self.guide_attn_conv(guide).view(batch_size, -1, w*h)
            attn = self.softmax(torch.bmm(src_attm, guide_attn))
            src_proj = self.src_conv(src).view(batch_size, num_channel, w*h).permute(0, 2, 1)

            out = torch.bmm(attn, src_proj)
            out = slef.output_conv(out) + src
        else:
            batch_size, num_channel, w, h = input.shape
            src_attn = self.src_attn_conv(input).view(batch_size, -1, w*h).permute(0, 2, 1).contiguous()
            guide_attn = self.guide_attn_conv(input).view(batch_size, -1, w*h).contiguous()
            attn = self.softmax(torch.bmm(src_attn, guide_attn))
            src_proj = self.src_conv(input).view(batch_size, -1, w*h).permute(0, 2, 1).contiguous()
            out = torch.bmm(attn, src_proj).permute(0, 2, 1).view(batch_size, -1, w, h).contiguous()
            out = self.output_conv(out) + input

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_nonlocal=True):
        super(Decoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 128, 'up')
        self.conv2 = conv5x5(128*3, 128)
        self.conv3 = conv5x5(128, 64, 'up')
        if use_nonlocal:
            self.nonlocal1 = NonlocalBlock(128, 4, False)
            self.nonlocal2 = NonlocalBlock(64, 4, False)
        self.conv4 = conv5x5(64*3+128, 64)
        self.conv5 = conv5x5(64, 32, 'up')
        self.conv6 = conv5x5(32*3+64, 32)
        self.conv7 = conv5x5(32, out_channels, "up", sigmoid=True)
        self.use_nonlocal = use_nonlocal


    def forward(self, input, skip, background_feat=None):
        if self.use_nonlocal:
            output = self.conv2(torch.cat([skip[0], self.nonlocal1(self.conv1(input))], 1))
            output = self.conv4(torch.cat([skip[1], self.nonlocal2(self.conv3(output)), background_feat[0]], 1))
        else:
            output = self.conv2(torch.cat([skip[0], self.conv1(input)], 1))
            output = self.conv4(torch.cat([skip[1], self.conv3(output), background_feat[0]], 1))
        output = self.conv6(torch.cat([skip[2], self.conv5(output), background_feat[1]], 1))
        '''
        output = self.conv1(torch.cat([skip[0], self.conv0(input)], 1))
        if self.use_nonlocal:
            output = self.nonlocal1(output)
        output = self.conv3(torch.cat([skip[1], self.conv2(output)], 1))
        if self.use_nonlocal:
            output = self.nonlocal2(output)
        output = self.conv5(torch.cat([skip[2], self.conv4(output), background_feat[0]], 1))
        output = self.conv7(torch.cat([self.conv6(output), background_feat[1]], 1))
        return self.conv8(output)
        '''
        return self.conv7(output)

class PoseEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PoseEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

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
        return output[-1], [output[5], output[3], output[0]], [output[5], output[3], output[1]]

class ForeGroundEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ForeGroundEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32*2, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64*2, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128*2, 256, "down")
        self.conv8 = conv5x5(256, 256)

    def forward(self, input, lateral): 
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(torch.cat([output[-1], lateral[-1]], 1)))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(torch.cat([output[-1], lateral[-2]], 1)))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(torch.cat([output[-1], lateral[-3]], 1)))
        output.append(self.conv8(output[-1]))
        return output[-1], [output[5], output[3], output[1]]

class MainEncoder(nn.Module):
    def __init__(self, pose_channels, foreground_channels):
        super(MainEncoder, self).__init__()
        self.pose_enc = PoseEncoder(pose_channels)
        self.fore_enc = ForeGroundEncoder(foreground_channels)
        #self.conv = conv1x1(512, 256)
        self.conv = BasicBlock(512, 256, downsample=nn.Sequential(
                conv1x1(512, 256),
                nn.BatchNorm2d(256),
            ))

    def forward(self, pose, foreground):
        pose_feat, lateral, pose_skip = self.pose_enc(pose)
        fore_feat, fore_skip = self.fore_enc(foreground, lateral)
        skip = list(map(lambda x:torch.cat(list(x), 1), zip(pose_skip, fore_skip)))
        feat = torch.cat([pose_feat, fore_feat], 1)
        feat = self.conv(feat)
        return feat, skip

class BackGroundEncoder(nn.Module):
    def __init__(self, in_channels, input_size, dataset='market1501'):
        super(BackGroundEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32)
        self.conv2 = conv5x5(32, 64, 'down')
        self.di_conv1 = conv3x3(64, 64, rate=2)
        self.di_conv2 = conv3x3(64, 64, rate=4)
        self.di_conv3 = conv3x3(64, 64, rate=8)
        '''
        self.conv3_1 = conv5x5(64, 64)
        '''
        self.conv3 = conv5x5(64, 128, 'down')
        size = (32, 16) if dataset == 'market1501' else (32, 32)
        self.upsample = nn.Upsample(size, mode='bilinear', align_corners=True)


    def forward(self, input):
        output = []
        x = self.conv1(input)
        x = self.conv2(x)
        output.insert(0, x)
        x = self.di_conv1(x)
        x = self.di_conv2(x)
        x = self.di_conv3(x)
        x = self.conv3(x)
        x = self.upsample(x)
        '''
        x = self.conv3_1(x)
        x = self.conv3(x)
        '''
        output.insert(0, x)
        #return x
        return output


class Pose_to_Image(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, input_size=128, 
        recurrent=1, use_nonlocal=True, dataset=None):
        super(Pose_to_Image, self). __init__()
        self.pose_channels = pose_channels
        self.img_channels = img_channels
        self.encoder_conv = MainEncoder(pose_channels, img_channels)
        self.decoder_conv = Decoder(256, 3, use_nonlocal=use_nonlocal)
        self.background_encoder = BackGroundEncoder(img_channels+1, input_size, dataset=dataset)
        self.recurrent = recurrent



    def forward(self, input):
        pose_imgs, human_imgs, background_imgs, masks = input
        #pose_human_imgs = torch.cat([pose_imgs, human_imgs], 1)
        foreground_feat, skip = self.encoder_conv(pose_imgs, human_imgs)
        outputs = []
        for i in range(self.recurrent):
            background_feat = self.background_encoder(torch.cat([background_imgs, masks], 1))
            #output, skip = self.encoder_conv(pose_human_imgs)
            output = self.decoder_conv(foreground_feat, skip, background_feat)
            background_imgs = output
            outputs.append(output)
        return outputs

class Model(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, input_size=128, 
        recurrent=1, use_nonlocal=True, dataset=None, use_gan=False, lr=2e-4,
        lambda1=0.5, lambda2=0.05, lambda3=0.5, mapping=None, roi_size=None, DVG=False, dual_path=True):
        super(Model, self).__init__()
        #main model
        if DVG:
            self.model = Pose_to_Image_DVG(pose_channels, img_channels, skip=True)
        elif not dual_path:
            self.model = Pose_to_Image_SinglePath(pose_channels, img_channels, input_size, recurrent, 
                use_nonlocal, dataset)
        else:
            self.model = Pose_to_Image(pose_channels, img_channels, input_size, recurrent,
                use_nonlocal, dataset)
        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.schd = optim.lr_scheduler.StepLR(self.optim, 50, 0.5)

        #for perceptual loss
        with torch.no_grad():
            self.vgg_layers = vgg.vgg19(pretrained=True).features

        #for gan
        if use_gan:
            from discriminator import GANLoss, ResnetDiscriminator
            self.crit_gan = GANLoss().cuda()
            self.net_d_pose = ResnetDiscriminator(input_nc=3+pose_channels).cuda()
            self.net_d_img = ResnetDiscriminator(input_nc=3+3).cuda()
            self.optim_d_pose = optim.Adam(self.net_d_pose.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
            self.optim_d_img = optim.Adam(self.net_d_img.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
            self.schd_d_pose = optim.lr_scheduler.StepLR(self.optim_d_pose, 5, 0.1)
            self.schd_d_img = optim.lr_scheduler.StepLR(self.optim_d_img, 5, 0.1)

        self.use_gan = use_gan
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.recurrent = recurrent
        self.layers_mapping = mapping
        self.roi_size = roi_size
        self.DVG = DVG

    def scheduler_step(self):
        self.schd.step()
        if self.use_gan:
            self.schd_d_pose.step()
            self.schd_d_img.step()

    def forward(self, input):
        self.input = input
        outputs = self.model(input)
        return outputs

    def cal_perc_feat(self, x, target_bbox=None):
        initial_size = x.size()
        image_w = initial_size[2]
        output = {}
        mask_output = {}
        roi_cnt = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layers_mapping:
                if target_bbox is not None:
                    spatial_scale = x.shape[2] / image_w
                    mask_output[self.layers_mapping[name]] = roi_pool(x, target_bbox, 
                        self.roi_size[roi_cnt], spatial_scale)
                    roi_cnt += 1
                output[self.layers_mapping[name]] = x
        return output, mask_output

    def cal_perc_loss(self, pred, target, roi_bbox=None):
        output = []
        perc_loss = []
        mask_perc_loss = []
        bs = pred.shape[0]
        feat, mask_feat = self.cal_perc_feat(torch.cat([pred, target], 0), roi_bbox)
        for k in feat.keys():
            perc_loss.append(torch.mean((feat[k][:bs]-feat[k][bs:]).pow(2)))
            if roi_bbox is not None:
                mask_perc_loss.append(torch.mean((mask_feat[k][:bs]-mask_feat[k][bs:]).pow(2)))
        perc_loss = torch.mean(torch.stack(perc_loss))
        mask_perc_loss = torch.mean(torch.stack(mask_perc_loss)) if roi_bbox is not None else None
        return perc_loss, mask_perc_loss

    def backward_g(self, pred, input, target, roi_bbox=None):
        self.optim.zero_grad()

        final_pred = pred[-1]
        pose_imgs, human_imgs, background_imgs, masks = input
        src_imgs = human_imgs + background_imgs
        perc_loss = []
        mask_perc_loss = []
        l1_loss = []
        for i in range(self.recurrent):
            pred_ = pred[i]
            l1_loss.append(nn.L1Loss()(pred_, target))
            #l1_loss.append(nn.MSELoss()(pred_, target))
            perc_loss_, mask_perc_loss_ = self.cal_perc_loss(pred_, target, roi_bbox)
            perc_loss.append(perc_loss_)
            if roi_bbox is not None:
                mask_perc_loss.append(mask_perc_loss_)

        l1_loss = torch.stack(l1_loss, 0).mean()
        perc_loss = torch.stack(perc_loss, 0).mean()

        g_loss = l1_loss + self.lambda1 * perc_loss
        #g_loss = self.lambda1 * (l1_loss + perc_loss)
        if len(mask_perc_loss) > 0:
            mask_perc_loss = torch.stack(mask_perc_loss, 0).mean() 
            g_loss = g_loss + self.lambda2 * mask_perc_loss 
        else:
            mask_perc_loss = None

        if self.use_gan:
            fake_pose_input = torch.cat([pose_imgs, final_pred], 1).detach()
            fake_img_input = torch.cat([src_imgs, final_pred], 1).detach()
            gan_loss = self.lambda3 * (self.crit_gan(self.net_d_pose(fake_pose_input), True) + 
                              self.crit_gan(self.net_d_img(fake_img_input), True))
            g_loss = g_loss + gan_loss
        else:
            gan_loss = None


        g_loss.backward()
        self.optim.step()

        return l1_loss, perc_loss, mask_perc_loss, gan_loss

    def backward_d_p(self, net_d, fake_input, real_input, optim_d):
        optim_d.zero_grad()
        d_loss = self.lambda3 * (self.crit_gan(net_d(real_input), True) +\
                        self.crit_gan(net_d(fake_input), False))
        d_loss.backward()
        optim_d.step()
        return d_loss

    def backward_d(self, pred, input, target):
        pose_imgs, human_imgs, background_imgs, masks = input
        src_imgs = human_imgs + background_imgs

        fake_pose_input = torch.cat([pose_imgs, pred[-1]], 1).detach()
        real_pose_input = torch.cat([pose_imgs, target], 1).detach()

        fake_img_input = torch.cat([src_imgs, pred[-1]], 1).detach()
        real_img_input = torch.cat([src_imgs, target], 1).detach()

        d_pose_loss = self.backward_d_p(self.net_d_pose, fake_pose_input, real_pose_input, self.optim_d_pose)
        d_img_loss = self.backward_d_p(self.net_d_img, fake_img_input, real_img_input, self.optim_d_img)

        d_loss = d_pose_loss + d_img_loss
        return d_loss

    def optimize(self, pose_imgs, human_imgs, background_imgs, masks, target_imgs, roi_bbox):
        input = (pose_imgs, human_imgs, background_imgs, masks)
        pred = self.forward(input)

        l1_loss, perc_loss, mask_perc_loss, gan_loss = self.backward_g(pred, input, target_imgs, roi_bbox)

        if self.use_gan:
            d_loss = self.backward_d(pred, input, target_imgs)
        else:
            d_loss = None

        return pred, l1_loss, perc_loss, mask_perc_loss, gan_loss, d_loss


