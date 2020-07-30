import os
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL
import PIL.Image as Image
from torchvision import transforms
import cv2
import sys
import pickle
from tqdm import tqdm
import json
import imageio

def kp_to_map(img_sz, kps, mode='binary', radius=5):
    '''
    Keypoint cordinates to heatmap map.
    Input:
        img_size (w,h): size of heatmap
        kps (N,2): (x,y) cordinates of N keypoints
        mode: 'gaussian' or 'binary'
        radius: radius of each keypoints in heatmap
    Output:
        m (h,w,N): encoded heatmap
    '''
    w, h = img_sz
    x_grid, y_grid = np.meshgrid(range(w), range(h), indexing = 'xy')
    m = []
    for x, y in kps:
        if x == -1 or y == -1:
            m.append(np.zeros((h, w)).astype(np.float32))
        else:
            if mode == 'gaussian':
                m.append(np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(radius**2)).astype(np.float32))
            elif mode == 'binary':
                m.append(((x_grid-x)**2 + (y_grid-y)**2 <= radius**2).astype(np.float32))
            else:
                raise NotImplementedError()
    m = np.stack(m, axis=0)
    return m

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.img_root = os.path.join(root, 'Images', 'img')
        self.pose_root = os.path.join(root, 'Images', 'pose')
        self.pose_img_root = os.path.join(root, mode, 'pose')
        self.mask_root = os.path.join(root, 'Images', 'seg')
        self.split = json.load(open(os.path.join(root, 'Label', 'pair_split.json')))[mode]
        self.poses = pickle.load(open(os.path.join(root, 'Label', 'pose_label.pkl'), 'rb'), encoding='latin1')
        t_list = [transforms.ToTensor(), 
                    lambda x: x[:3, ::],
                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
        if 'market' in root:
            t_list.insert(0, transforms.Scale([128, 64]))
        else:
            t_list.insert(0, transforms.Scale([128, 128]))
        self.transforms = transforms.Compose(t_list)
        self.mask_transforms = transforms.Compose([
        lambda x: torch.Tensor(x),
    ])



    def __getitem__(self, item):
        src, target = self.split[item]
        mask = imageio.imread(os.path.join(self.mask_root, '{}.bmp'.format(src)))
        mask = (mask != 0).astype(int)

        src_img_path = os.path.join(self.img_root, '{}.jpg'.format(src))
        target_img_path = os.path.join(self.img_root, '{}.jpg'.format(target))
        
        mask  = torch.FloatTensor(mask).unsqueeze(0)

        target_pose = self.poses[target]

        target_mask_path = os.path.join(self.mask_root, '{}.bmp'.format(target))
        target_mask = imageio.imread(target_mask_path)
        target_mask = (target_mask != 0).astype(int)
        left_x = np.min(np.where(target_mask!=0)[0])
        left_y = np.min(np.where(target_mask!=0)[1])
        right_x = np.max(np.where(target_mask!=0)[0])
        right_y = np.max(np.where(target_mask!=0)[1])
        target_mask = np.zeros([1, target_mask.shape[0], target_mask.shape[1]])
        target_mask[:, left_x:right_x+1, left_y:right_y+1] = 1
        target_mask = torch.FloatTensor(target_mask)
        src_img_mat = self.transforms(Image.open(src_img_path))
        human_img = src_img_mat * mask
        background_img = src_img_mat * (1 - mask)
        src_img = self.transforms(Image.open(src_img_path))
        masked_idx = (target_mask[0, :, :]!=0).nonzero()
        try:
            min_x = torch.min(masked_idx[:, 0]).item()
        except RuntimeError:
            import pdb;pdb.set_trace()
        min_y = torch.min(masked_idx[:, 1]).item()
        max_x = torch.max(masked_idx[:, 0]).item()
        max_y = torch.max(masked_idx[:, 1]).item()
        roi_bbox = torch.FloatTensor([min_x, min_y, max_x, max_y])
        target_img = self.transforms(Image.open(target_img_path))
        pose_img = kp_to_map((target_img.shape[2], target_img.shape[1]), target_pose)
        if self.mode == 'train':
            return pose_img, target_img, human_img, background_img, mask, target_mask, roi_bbox
        else:
            return pose_img, target_img, human_img, background_img, mask, target_mask, roi_bbox, src, target
            

    def __len__(self):
        return len(self.split)
