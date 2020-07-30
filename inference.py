from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import os
import PIL.Image as Image
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import datasets
import sys
import cv2
#from model_decompose import Pose_to_Image, Perceptual
#from model_DVG import Pose_to_Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="name of model")
parser.add_argument('-dr', '--root', default='/home/xuchengming/XCM/Datasets', help="data root")
parser.add_argument('-d', '--dataset', help='dataset name')
parser.add_argument('-r', '--recurrent', type=int, default=3, help='num recurrent')
parser.add_argument('-nl', '--non_local', action='store_true', help='non local')
parser.add_argument('-dp', '--dual_path', action='store_true', help='dual path')
parser.add_argument('-sum', '--sum', action='store_true', help='dual path')
parser.add_argument('-gan', '--use_gan', action='store_true', help='use non local block')
parser.add_argument('-DVG', '--DVG', action='store_true', help='use non local block')
args = parser.parse_args()
from model_decompose_nlblockDecoder import Model
if args.dataset == 'market1501':
    pose_channels = 18 
elif args.dataset == 'Penn_Action':
    pose_channels = 13
else:
    pose_channels = 7
model = Model(pose_channels, 3, recurrent=args.recurrent,
                use_nonlocal=args.non_local, dataset=args.dataset,
                use_gan=args.use_gan, DVG=args.DVG, dual_path=args.dual_path).to('cuda')


if args.name not in ['src', 'target', 'mask']:
    all_ckpt = list(filter(lambda x:x.startswith(args.dataset), os.listdir(os.path.join('./models', args.name))))
    index = max(list(map(lambda x:int(x.split('.')[0].split('_')[-1]), all_ckpt)))
    model_path = os.path.join('./models', args.name, '{}_{}.pt'.format(args.dataset, index))
    print('using model {}'.format(model_path))
    state_dict = torch.load(model_path)
    state_dict =  {k.replace('module', 'model'):v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
model = nn.DataParallel(model)
model.eval()
#writer = SummaryWriter('./results' )
test_data = datasets.PoseDataset(os.path.join(args.root, args.dataset), mode='test')

test_loader = DataLoader(test_data,
                          num_workers=0,
                          batch_size=10,
                          shuffle=False,
                          pin_memory=True)

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

def get_img_batch(batch):
    images = []
    for i in range(batch.shape[0]):
        image = batch[0].data.cpu()
        norm_ip(image)
        images.append(image.mul(255).clamp(0, 255).permute(1, 2, 0).numpy())
    return images

result_root = os.path.join('./result_images/', args.dataset,args.name)

if not os.path.isdir(result_root):
    os.makedirs(result_root, exist_ok=True)
file_dict = dict()
for j, (pose_imgs, target_imgs, human_imgs, background_imgs, masks, target_masks, roi_bbox, src_name, target_name) in enumerate(test_loader):
    print(j)
    if torch.cuda.is_available():
        pose_imgs = Variable(pose_imgs).cuda()
        target_imgs = Variable(target_imgs).cuda()
        human_imgs = Variable(human_imgs).cuda()
        background_imgs = Variable(background_imgs).cuda()
        masks = Variable(masks).cuda()

    src_imgs = human_imgs + background_imgs
    #pred = model(torch.cat([pose_imgs, src_imgs], 1))
    pred = model((pose_imgs, human_imgs, background_imgs, masks))
    pred = pred[-1]
    #os.makedirs(os.path.join(result_root, '%d' % j), exist_ok=True)
    for k in range(pose_imgs.shape[0]):
        if args.name == 'src':
            pred_img = src_imgs[k].data.cpu()
        elif args.name == 'target':
            pred_img = target_imgs[k].data.cpu()
        elif args.name == 'mask':
            pred_img = masks[k].data.cpu()
        else:
            pred_img = pred[k].data.cpu()
        norm_ip(pred_img, torch.min(pred_img), torch.max(pred_img))
        cv2.imwrite(os.path.join(result_root, "{}_{}.png".format(src_name[k], target_name[k])), pred_img.mul(255).clamp(0, 255).numpy().transpose(1, 2, 0)[..., ::-1])
'''
import pickle
w = open("Market-1501_origin_order.pkl", 'wb')
pickle.dump(file_dict, w)
w.close()
'''
