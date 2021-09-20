import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from tqdm import tqdm
from os import listdir
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from random import sample
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics

def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img, 0)
        # backward compatibility
        return img.float()


def NII_loader(img, data):
    nii_img = nib.load(img).get_fdata()
    #  MRI normalize
    image_array = np.array(nii_img)
    max_num = np.max(image_array)
    min_num = np.min(image_array)
    image_array = (image_array - min_num)/(max_num - min_num)

    data = np.array(data)

    # To Tensor
    image_array = customToTensor(image_array)
    data = torch.from_numpy(data).float()
    return image_array, data


class AD_Dataset(Dataset):
    def __init__(self, transform=None, loader=NII_loader, dataset_id=None, id_table=None):
        super(AD_Dataset, self).__init__()
        x_train = []
        clinical_data = []
        y_train = []
        conversion_table = id_table

        AD_path = '/AD_subjects/'

        AD_files = os.listdir(AD_path)

        for file in AD_files:
            files = AD_path + file
            match_id = re.search(r'[0-9][0-9][0-9]_S_[0-9][0-9][0-9][0-9]', file)
            if [match_id[0]] in dataset_id:
                if conversion_table[match_id[0]][0] != 2:
                    x_train.append(files)
                    clinical_data.append(conversion_table[match_id[0]][1:])
                    y_train.append(conversion_table[match_id[0]][0])

        # data normalize
        data = np.array(clinical_data)
        high = np.max(data, axis=0)
        low = np.min(data, axis=0)
        div = high - low

        for i in range(div.shape[1]):
            if div[0][i] == 0:
                div[0][i] = 1

        data = (data - low) / (div)

        self.x_train = x_train
        self.CLI_train = data.tolist()
        self.y_train = y_train
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        N_img, CLI_data, label = self.x_train[index], self.CLI_train[index][0], self.y_train[index]
        N_img, CLI_data = self.loader(N_img, CLI_data)
        return N_img, CLI_data, label

    def __len__(self):
        return len(self.x_train)