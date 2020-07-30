import numpy as np
import os
from skimage.measure import compare_ssim
import skimage.io as sio
from skimage import transform
import cv2
import pickle
from skimage.draw import circle, line_aa, polygon

def produce_ma_mask(kp_array, point_radius=4):
    from skimage.morphology import dilation, erosion, square
    MISSING_VALUE = -1
    img_size = [128, 64]
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 1], vetexes[:, 0], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[1], joint[0], radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask.astype(int)

def ssim(img1, img2):
    return compare_ssim(img1, img2, multichannel=True, win_size=11)

dataset = 'Penn_Action'
dataset = 'market1501'
root = '/home/xuchengming/XCM/Previous-Works/poseguide-TIP2020/result_images'
target_root = os.path.join(root, dataset, 'target')
models = os.listdir(os.path.join(root, dataset))
seg_root = '/home/xuchengming/XCM/Datasets/{}/Images/seg'.format(dataset)

file_list = os.listdir(os.path.join(root, dataset, 'target'))
all_poses = pickle.load(open(os.path.join('/home/xuchengming/XCM/Datasets/{}/Label/pose_label.pkl').format(dataset), 'rb'), encoding='latin1')
import pickle
scores = {}
masked_scores = {}
L1 = {}
masked_L1 = {}
for model in models:
    scores[model] = list()
    masked_scores[model] = list()

for idx, img in enumerate(file_list):
    if idx % 1000 == 0:
        print('%d images have been evaluated' % idx)
    target_img = sio.imread(os.path.join(target_root, img))
    items = img.replace('.jpg','').replace('.png', '').split('_')
    target = '_'.join(items[int(len(items)/2):])
    try:
        if dataset == 'market1501':
            pose_img = produce_ma_mask(all_poses[target])
        else:
            pose_img = sio.imread(os.path.join(seg_root, target+'.bmp'))
    except KeyError:
        if dataset == 'market1501':
            pose_img = produce_ma_mask(all_poses[target+'.jpg'])
        else:
            continue
            
    if len(pose_img.shape) == 2:
        pose_img = np.stack([pose_img, pose_img, pose_img], -1)
    crop_target = target_img * pose_img
    for model in models:
        if os.path.isfile(os.path.join(root, dataset, model, img)):
            pred_img = sio.imread(os.path.join(root, dataset, model, img))
        else:
            pred_img = sio.imread(os.path.join(root, dataset, model, img.split('.')[0]+'.jpg'))
        crop_pred = pred_img * pose_img
        ssim = compare_ssim(target_img, pred_img, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, multichannel=True,
                                                                data_range=pred_img.max() - pred_img.min())


        scores[model].append(ssim)
        mask_ssim = compare_ssim(crop_target, crop_pred, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, multichannel=True,
                                                                data_range=crop_pred.max() - crop_pred.min())
        masked_scores[model].append(mask_ssim)

for model in models:
    print(model)
    print(np.mean(scores[model]))
    print(np.mean(masked_scores[model]))
