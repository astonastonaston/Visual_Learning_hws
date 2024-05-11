import torch
import PIL
from torch.utils.data import Dataset
import os
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


def get_full_list(
    root_dir,
    base_dir="base",
    extended_dir="extended",
):
    data_list = []
    for name in [base_dir, extended_dir]:
        data_dir = osp.join(
            root_dir, name
        )
        data_list += sorted(
            osp.join(data_dir, img_name) for img_name in
            filter(
                lambda x: x[-4:] == '.jpg',
                os.listdir(data_dir)
            )
        )
    return data_list

class CMP_Facade_DB(Dataset):
    def __init__(
        self,
        data_list
    ):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
                
    def __getitem__(self, i):
        # input and target images
        in_name = self.data_list[i]
        gt_name = self.data_list[i].replace('.jpg','.png')
    
        # process the images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transf_img = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        in_image = Image.open(in_name).convert('RGB')
        gt_label = Image.open(gt_name)
        w, h = in_image.size
        gt_label = np.frombuffer(gt_label.tobytes(), dtype=np.ubyte).reshape((h, w))
        
        in_image = cv2.resize(np.array(in_image), None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
        gt_label = cv2.resize(np.array(gt_label), None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
        
        in_image = transf_img(in_image)
        gt_label = torch.tensor(gt_label)
        gt_label = (gt_label).long() - 1

        return in_image, gt_label
    
    def revert_input(self, img, label):
        img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        std_img = np.array([0.229, 0.224, 0.225]).reshape((1, 1, -1))
        mean_img = np.array([0.485, 0.456, 0.406]).reshape((1, 1, -1))
        img *= std_img
        img += mean_img
        label = label.cpu().numpy()
        return img, label + 1
