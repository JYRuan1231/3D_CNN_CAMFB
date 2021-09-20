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

from model import *
from dataset import *



def evaluate(label, pred, preds,  dataset_size):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = metrics.auc(fpr, tpr)

    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    spe = 1-fpr[ix]
    sen = tpr[ix]

    preds = np.array(preds)
    label = np.array(label)

    thrs = [best_thresh]
    preds[preds < thrs[0]] = 0.0
    preds[preds >= thrs[0]] = 1.0

    running_corrects = np.sum(preds == label)

    epoch_acc = running_corrects / dataset_size
    epoch_sen = sen
    epoch_spe = spe

    return epoch_acc, epoch_sen, epoch_spe, auc



def train_evaluate_model(model, criterion, optimizer, scheduler, num_epochs=25):

    epoch_loss = 0.0
    epoch_acc = 0.0
    accumulation_steps = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        process = ['train', 'val', 'test']

        # Each epoch has a training and validation phase
        for phase in process:
            if phase == 'train':
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = len(train_loader)
            elif phase == 'val':
                model.eval()  # Set model to validation mode
                model_loader = valid_loader
                dataset_size = len(valid_loader)
            elif phase == 'test':
                model.eval()  # Set model to testing mode
                model_loader = test_loader
                dataset_size = len(test_loader)

            running_loss = 0.0

            preds = []
            train_pred = []
            train_label = []
            valid_pred = []
            valid_label = []
            test_label = []
            test_pred = []

            # Iterate over data.
            if phase == 'train':
                for iter, (imgs, data, labels) in tqdm(enumerate(model_loader), total=dataset_size):

                    imgs = imgs.to(device)
                    data = data.to(device)
                    labels = labels.to(device)

                    outputs = model(imgs, data)

                    rg_outputs = outputs
                    rg_outputs = torch.sigmoid(rg_outputs)

                    rg_loss = criterion['regression'](rg_outputs.view(-1), labels.float())

                    loss = rg_loss
                    loss = loss / accumulation_steps

                    loss.backward()
                    if (iter + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    train_pred += rg_outputs.tolist()
                    train_label += labels.tolist()
                    # statistics
                    preds += rg_outputs.unsqueeze(1)
                    running_loss += loss.item() * imgs.size(0)

                scheduler.step()
                epoch_loss = running_loss / dataset_size / batch_size
                epoch_acc, epoch_sen, epoch_spe, auc = evaluate(train_label, train_pred, preds, dataset_size * batch_size)


            elif phase == 'val':
                with torch.no_grad():
                    for iter, (imgs, data, labels) in tqdm(enumerate(model_loader), total=dataset_size):
                        imgs = imgs.to(device)
                        data = data.to(device)
                        labels = labels.to(device)

                        outputs = model(imgs, data)

                        rg_outputs = outputs
                        rg_outputs = torch.sigmoid(rg_outputs)

                        rg_loss = criterion['regression'](rg_outputs.view(-1), labels.float())
                        loss = rg_loss
                        loss = loss / accumulation_steps
                        valid_pred += rg_outputs.tolist()
                        valid_label += labels.tolist()

                        # statistics
                        preds += rg_outputs.unsqueeze(1)
                        running_loss += loss.item() * imgs.size(0)
                    epoch_loss = running_loss / dataset_size
                    epoch_acc, epoch_sen, epoch_spe, auc = evaluate(valid_label, valid_pred, preds, dataset_size)

            elif phase == 'test':
                with torch.no_grad():
                    for iter, (imgs, data, labels) in tqdm(enumerate(model_loader), total=dataset_size):
                        imgs = imgs.to(device)
                        data = data.to(device)
                        labels = labels.to(device)

                        # forward
                        # track history if only in train
                        outputs = model(imgs, data)

                        rg_outputs = outputs[0]
                        rg_outputs = torch.sigmoid(rg_outputs)

                        rg_loss = criterion['regression'](rg_outputs.view(-1), labels.float())

                        loss = rg_loss
                        loss = loss / accumulation_steps
                        test_pred += rg_outputs.tolist()
                        test_label += labels.tolist()

                        # statistics
                        preds += rg_outputs.unsqueeze(1)
                        running_loss += loss.item() * imgs.size(0)
                epoch_loss = running_loss / dataset_size
                epoch_acc, epoch_sen, epoch_spe, auc = evaluate(test_label, test_pred, preds, dataset_size)

            print('{} Loss: {:.4f} Acc: {:.4f} Sen: {:.4f} Spe: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_sen, epoch_spe, auc))
        print()
        print("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

if __name__ == "__main__":
    """Setting args"""

    learning_rate = 5e-4
    batch_size = 6
    num_epochs = 50
    mci_path = 'DXSUM_pMCI.csv'
    ad_subjuct_path = './ADNIMERGE_BL_ADNI1.csv'

    # Statistics AD subjects
    class_stats = [0, 0, 0, 0]
    conversion_table = {}
    ad_list = []
    nc_list = []
    pmci_list = []
    smci_list = []
    ad_num = 0
    nc_num = 0
    pmci_num = 0
    smci_num = 0

    # The ID list of AD subjects
    with open(ad_subjuct_path, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            nan_signal = 0
            age = row['AGE']
            if row['PTGENDER'] == 'Male':
                gender = 1
            else:
                gender = 0
            edu = row['PTEDUCAT']
            apoe = row['APOE4']
            cdrs = row['CDRSB']
            ad11 = row['ADAS11']
            ad13 = row['ADAS13']
            adq4 = row['ADASQ4']
            mmse = row['MMSE']
            ri = row['RAVLT_immediate']
            rl = row['RAVLT_learning']
            rf = row['RAVLT_forgetting']
            rp = row['RAVLT_perc_forgetting']
            ldtotal = row['LDELTOTAL']
            digit = row['DIGITSCOR']
            cognitive_score = [age, gender, edu, apoe, cdrs, ad11, ad13, ri, rl, rf, rp]
            for i in cognitive_score:
                if i == '':
                    print(row['PTID'])
                    nan_signal = 1
            if nan_signal == 1:
                continue
            cognitive_score = [float(i) for i in cognitive_score]
            PTID = row['PTID']
            if row['DX_bl'] == 'CN':
                conversion_table[PTID] = [0, cognitive_score]
                nc_list.append([PTID])
                class_stats[0] += 1

            if row['DX_bl'] == 'AD':
                conversion_table[PTID] = [1, cognitive_score]
                ad_list.append([PTID])
                class_stats[1] += 1

            if row['DX_bl'] == 'LMCI':
                f = open(mci_path, 'r', newline='')
                rows = csv.DictReader(f)
                conversion_table[PTID] = [0, cognitive_score]
                smci_list.append([PTID])
                class_stats[0] += 1
                for row in rows:
                    if row['PTID'] == PTID:
                        if row['VISCODE'] == 'm48':
                            break
                        conversion_table[PTID] = [1, cognitive_score]
                        pmci_list.append([PTID])
                        smci_list.pop()
                        class_stats[1] += 1
                        class_stats[0] -= 1
                        break
    # 1 times for training and testing model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_smci = list(smci_list)
    train_pmci = list(pmci_list)

    test_smci = sample(train_smci, 16)
    test_pmci = sample(train_pmci, 16)

    for i in test_smci:
        if i in train_smci:
            train_smci.remove(i)

    for i in test_pmci:
        if i in train_pmci:
            train_pmci.remove(i)

    train_list = train_smci + train_pmci

    for_valid_mci = sample(train_list, 36)

    for i in for_valid_mci:
        if i in train_list:
            train_list.remove(i)

    train_list = train_list
    valid_list = for_valid_mci
    test_list = test_smci + test_pmci
    print('train_list:', len(train_list))
    print('valid_list:', len(valid_list))
    print('test_list:', len(test_list))

    train_dataset = AD_Dataset(dataset_id=train_list, id_table=conversion_table)
    valid_dataset = AD_Dataset(dataset_id=valid_list, id_table=conversion_table)
    test_dataset = AD_Dataset(dataset_id=test_list, id_table=conversion_table)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    criterion = {
        'regression': nn.BCELoss()
    }

    camfb_model = CNN_with_CAMFB()
    camfb_model = camfb_model.to(device)

    optimizer_ft = optim.Adam(camfb_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=1e-5,amsgrad=False)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=50, T_mult=2, eta_min=1e-5)

    train_evaluate_model(camfb_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

