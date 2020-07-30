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
class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_pair=None):
        self.root = root
        self.pose_root = os.path.join(root, 'pose')
        self.src_root = os.path.join(root, 'src')
        self.target_root = os.path.join(root, 'target')
        self.mask_root = os.path.join(root, 'masks')
        # self.video_list = os.listdir(self.src_root)
        self.file_list = []
        for _, video in tqdm(enumerate(os.listdir(self.pose_root))):
            pose_list = list(map(lambda x:os.path.join(video, x), os.listdir(os.path.join(self.pose_root, video))))
            # pose_list = list(filter(lambda x:'cropout' not in x, pose_list))
            self.file_list += pose_list
            if max_pair is not None and len(self.file_list) >= max_pair:
                break
        #w = open("Market_inference_order.pkl", 'wb')

        #pickle.dump(self.file_list, w)
        #sys.exit()

        self.transforms = transforms.Compose([
            transforms.Scale([128, 128]),
            transforms.ToTensor(),
        lambda x: x[:3, ::],
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])
        self.split = root.split('/')[-1]

    def __getitem__(self, item):
        file_chosen = self.file_list[item]
        video = file_chosen.split('/')[0]
        src_img_path = list(filter(lambda x:'cropout' not in x, os.listdir(os.path.join(self.src_root, video))))[0]
        mask = src_img_path.split('.')[0]+'.npy'
        mask = np.load(os.path.join(self.mask_root, video, mask))
        comp_mask = np.stack([mask, mask, mask]).transpose(1, 2, 0)
        comp_mask = comp_mask.astype(np.uint8)
        try:
            mask  = self.transforms(Image.fromarray(comp_mask))
        except TypeError:
            import pdb;pdb.set_trace()
        mask = (mask != -1)[0, :, :].float().unsqueeze(0)

        pose_img = Image.open(os.path.join(self.pose_root, file_chosen))
        pose_mat = np.asarray(pose_img)
        left_x = np.min(np.where(pose_mat!=0)[0])
        left_y = np.min(np.where(pose_mat!=0)[1])
        right_x = np.max(np.where(pose_mat!=0)[0])
        right_y = np.max(np.where(pose_mat!=0)[1])
        target_mask = np.zeros_like(pose_mat)
        target_mask[left_x:right_x+1, left_y:right_y+1, :] = 1
        target_mask = self.transforms(Image.fromarray(target_mask))[0, :, :]
        target_mask = (target_mask != -1).float().unsqueeze(0)
        src_img_mat = np.asarray(Image.open(os.path.join(self.src_root, video, src_img_path)))
        human_img = self.transforms(Image.fromarray(src_img_mat*comp_mask))
        background_img = self.transforms(Image.fromarray(src_img_mat*(1-comp_mask)))
        src_img = self.transforms(Image.open(os.path.join(self.src_root, video, src_img_path)))
        #pose_img = self.transforms(Image.open(os.path.join(self.pose_root, file_chosen)))
        pose_img = self.transforms(pose_img)
        #target_mask = (pose_img != -1).float()
        masked_idx = (target_mask[0, :, :]!=0).nonzero()
        min_x = torch.min(masked_idx[:, 0]).item()
        min_y = torch.min(masked_idx[:, 1]).item()
        max_x = torch.max(masked_idx[:, 0]).item()
        max_y = torch.max(masked_idx[:, 1]).item()
        roi_bbox = torch.FloatTensor([min_x, min_y, max_x, max_y])
        #target_img = target_mask * np.asarray(Image.open(os.path.join(self.target_root, file_chosen)))
        #target_img = self.transforms(Image.fromarray(target_img))
        target_img = self.transforms(Image.open(os.path.join(self.target_root, file_chosen)))
        if self.split == 'test':
            return pose_img, target_img, human_img, background_img, mask, target_mask, roi_bbox, '1', '1'
        return pose_img, target_img, human_img, background_img, mask, target_mask, roi_bbox

    def __len__(self):
        return len(self.file_list)
