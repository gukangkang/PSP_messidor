from __future__ import print_function, division, absolute_import
import numpy as np
import os
# Ignore warnings
import warnings

import scipy.misc as misc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize,Scale
from torchvision.transforms import ToTensor, ToPILImage
warnings.filterwarnings('ignore')
from transform import Relabel, ToLabel, Colorize
import scipy.io as scio
import torch


def preprocessed_mat(image_path):
    data = scio.loadmat(image_path)
    # print(type(data))
    #
    # print(data.keys())
    # original_data = data['cupmask']
    original_data = data['cropped']
    original_data = original_data
    return original_data


class OrigaDataset(Dataset):
    """
    Define the ORIGA dataset for Pytorch
    """

    def __init__(self, Set_A_dir, image_root, mask_root, input_transform=None, target_transform=None):
        image_names = []
        image_name = open(Set_A_dir).readlines()
        for image in image_name:
            image_names.append(image)

        self.image_name = image_names
        self.image_root_dir = image_root
        self.mask_root_dir = mask_root
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_root_dir, self.image_name[idx][:-1])
        mask_name = os.path.join(self.mask_root_dir, self.image_name[idx][:-4]+'mat')

        img = misc.imread(img_name)
        mask = preprocessed_mat(mask_name)
        # mask_onehot = np.eye(3)[mask]
        # print(np.shape(mask_onehot))

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # print(mask_onehot.size())

        sample = {'image': img, 'mask': mask}

        return sample


if __name__ == '__main__':

    txt_dir='/home/imed/workspace/PycharmProjects/pspnet-pytorch-master/Source/Set_A.txt'
    image_folder = '/home/imed/workspace/PycharmProjects/pspnet-pytorch-master/Source/rename_image'
    mask_folder = '/home/imed/workspace/PycharmProjects/pspnet-master/Source/rename_gt'

    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])

    glaucoma = OrigaDataset(Set_A_dir=txt_dir,
                            image_root=image_folder,
                            mask_root=mask_folder,
                            input_transform=input_transform,
                            target_transform=target_transform
                            )

    #######################
    # test the Dataset
    #######################

    loader = DataLoader(glaucoma, num_workers=4, batch_size=5)
    print(len(loader))
    # for i in range(len(loader)):
    #     sample = glaucoma[i]
    #
    #     print(i, sample['image'].size(), sample['mask'].size())
    #
    #     if i == 10:
    #         break

    for i_batch, sample_batch in enumerate(loader):
        print(i_batch, sample_batch['image'].size(), sample_batch['mask'].size())
        # print(torch.transpose(sample_batch['mask'], 1, 3).max())

        if i_batch == 4:
            break
