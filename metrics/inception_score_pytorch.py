import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import PIL.Image as Image

inception_model = inception_v3(pretrained=True, transform_input=False).to('cuda')
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = os.listdir(path)
        self.path = path
        self.transform = transform
        
    def __getitem__(self, index):
        x = Image.open(os.path.join(self.path, self.image_paths[index]))
        
        if self.transform:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.image_paths)
    

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    image_transforms = transforms.Compose([
                transforms.Scale([128, 128]),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                 ])

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=4)

    # Load inception model
    num_classes = 1000 #15 for penn action f.t.
    #inception_model.fc = nn.Linear(inception_model.fc.in_features, num_classes).cuda()
    #inception_model.load_state_dict(torch.load('../finetune_inception/model/Penn_epoch60.pt'))
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, 1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, num_classes))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    image_transforms = transforms.Compose([
        #transforms.Scale([128, 128]),
            transforms.ToTensor(),
                lambda x: x[:3, ::],
                    transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
                    ])

    #dataset = 'Penn_Action'
    #dataset = 'bbcpose'
    dataset = 'market1501'
    root = '/home/xuchengming/XCM/Previous-Works/poseguide-TIP2020_bk/result_images/'
    models = os.listdir(os.path.join(root, dataset))
    for model in models:
        selfdata = MyDataset(os.path.join(root, dataset, model), transform=image_transforms)
        print(model)
        print (inception_score(selfdata, cuda=True, batch_size=32, resize=True, splits=10))
