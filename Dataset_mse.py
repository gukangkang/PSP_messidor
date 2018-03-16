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
from utils import Constants
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

    def __init__(self, data_type=Constants.TYPE_TRAIN,
                 input_transform=None, target_transform=None):

        self.data_type = data_type
        self.image_root_dir = Constants.IMAGE_PATH
        self.mask_root_dir = Constants.MASK_PATH
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.train_files = []
        self.train_labels = []
        self.test_files = []
        self.test_labels =[]

        for image in open(Constants.ORIGA_TRAIN_FILE).readlines():
            self.train_files.append(image[:23]+'png')
            self.train_labels.append(image[:23]+'png')

        for image in open(Constants.ORIGA_TEST_FILE).readlines():
            self.test_files.append(image[:23] + 'png')
            self.test_labels.append(image[:23] + 'png')


    def __getitem__(self, idx):
        if self.data_type == Constants.TYPE_TRAIN:
            name = self.train_files[idx]
            img_name = os.path.join(self.image_root_dir, self.train_files[idx])
            mask_name = os.path.join(self.mask_root_dir, self.train_labels[idx])

        else:
            name = self.test_files[idx]
            img_name = os.path.join(self.image_root_dir, self.test_files[idx])
            mask_name = os.path.join(self.mask_root_dir, self.test_labels[idx])
        
        image_name = name
        img = misc.imread(img_name)
        mask = misc.imread(mask_name)
        # mask_onehot = np.eye(3)[mask]
        # print(np.shape(mask_onehot))

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # print(mask_onehot.size())

        sample = {'image': img, 'mask': mask, 'name': image_name}

        return sample

    def __len__(self):
        if self.data_type == Constants.TYPE_TRAIN:
            paths = self.train_files
        else:
            paths = self.test_files

        return len(paths)

if __name__ == '__main__':

    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])

    glaucoma = OrigaDataset(input_transform=input_transform,
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
