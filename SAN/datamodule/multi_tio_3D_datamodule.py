# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import torchio as tio
from utils.util import find_full_name
from torchvision.utils import save_image
import numpy as np
import torch

def image_reader(path):
    data = np.load(path)
    data = torch.from_numpy(data).float()
    data /= 255.
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine

def label_reader(path):
    data = np.load(path)
    data = torch.from_numpy(data).long()
    data //= 255
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine
class MultiTioDatamodule(pl.LightningDataModule):
    def __init__(self, image_root, label_root, region_mask_root, train_data_list, test_data_list, batch_size,
                 num_workers, train_per_size, queue_length, samples_per_volume, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_root
        self.label_root = label_root
        self.region_mask_root = region_mask_root
        self.train_names = train_data_list
        self.test_names = test_data_list

        self.patch_size = train_per_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume

    def prepare_data(self):
        self.train_subjects = []
        self.test_subjects = []
        for name in self.train_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, find_full_name(self.image_root, name)),
                                      reader=image_reader),
                label=tio.LabelMap(os.path.join(self.label_root, find_full_name(self.image_root, name)),
                                   reader=image_reader),
                region_mask=tio.LabelMap(os.path.join(self.region_mask_root, find_full_name(self.image_root, name)),
                                         reader=image_reader),
                name=name
            )
            subject.load()
            self.train_subjects.append(subject)
        for name in self.test_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, find_full_name(self.image_root, name)),
                                      reader=image_reader),
                label=tio.LabelMap(os.path.join(self.label_root, find_full_name(self.image_root, name)),
                                   reader=image_reader),
                region_mask=tio.LabelMap(os.path.join(self.region_mask_root, find_full_name(self.image_root, name)),
                                         reader=image_reader),
                name=name
            )
            # print(name)
            subject.load()
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2)),
        ])
        return augment

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        transform = tio.Compose([preprocess, augment])
        train_set = tio.SubjectsDataset(self.train_subjects, transform=transform)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

        self.patch_train_set = tio.Queue(
            train_set,
            self.queue_length,
            self.samples_per_volume,
            tio.data.UniformSampler(self.patch_size),
            num_workers=self.num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )

    def train_dataloader(self):
        return DataLoader(self.patch_train_set, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=None, batch_sampler=None)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=None, batch_sampler=None)


class MultiTioDatamodule_sunweifeng(pl.LightningDataModule):
    def __init__(self, image_root_sun, batch_size,
                 num_workers, train_per_size, queue_length, samples_per_volume, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_root_sun
        self.test_names =  os.listdir(image_root_sun)
        print(self.test_names)

        self.patch_size = train_per_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume

    def prepare_data(self):
        self.test_subjects = []
        for name in self.test_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, find_full_name(self.image_root, name)),
                                      reader=image_reader),
                name=name
            )
            # print(name)
            # subject.load()
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=None, batch_sampler=None)


class MultiTioDatamodule_v2(pl.LightningDataModule):
    def __init__(self, image_root, label_root, region_mask_root, train_data_list, test_data_list, batch_size,
                 num_workers, train_per_size, queue_length, samples_per_volume, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_root
        self.label_root = label_root
        self.region_mask_root = region_mask_root
        self.train_names = train_data_list
        self.test_names = test_data_list

        self.patch_size = train_per_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume

    def prepare_data(self):
        self.train_subjects = []
        self.test_subjects = []
        for name in self.train_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, find_full_name(self.image_root, name)),
                                      reader=image_reader),
                label=tio.LabelMap(os.path.join(self.label_root, find_full_name(self.image_root, name)),
                                   reader=label_reader),
                region_mask=tio.LabelMap(os.path.join(self.region_mask_root, find_full_name(self.image_root, name)),
                                         reader=label_reader),
                name=name
            )
            subject.load()
            self.train_subjects.append(subject)
        for name in self.test_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, find_full_name(self.image_root, name)),
                                      reader=image_reader),
                label=tio.LabelMap(os.path.join(self.label_root, find_full_name(self.image_root, name)),
                                   reader=image_reader),
                region_mask=tio.LabelMap(os.path.join(self.region_mask_root, find_full_name(self.image_root, name)),
                                         reader=image_reader),
                name=name
            )
            # print(name)
            subject.load()
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2)),
        ])
        return augment

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        transform = tio.Compose([preprocess, augment])
        train_set = tio.SubjectsDataset(self.train_subjects, transform=transform)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

        self.patch_train_set = tio.Queue(
            train_set,
            self.queue_length,
            self.samples_per_volume,
            tio.data.UniformSampler(self.patch_size),
            num_workers=self.num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )

    def train_dataloader(self):
        return DataLoader(self.patch_train_set, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=None, batch_sampler=None)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=None, batch_sampler=None)

if __name__ == '__main__':
    OCT_npy_root = '/home/Data/huangkun/Choroid/DATA/dataset4/npys/OCT'
    label_npy_root = '/home/Data/huangkun/Choroid/DATA/dataset4/npys/labels'
    region_mask_npy_root = '/home/Data/huangkun/Choroid/DATA/dataset4/npys/region_mask'
    train_data_list = ['10001', '10056', '10164']
    test_data_list = ['10001', '10056', '10164']
    dm = MultiTioDatamodule(OCT_npy_root, label_npy_root, region_mask_npy_root, train_data_list, test_data_list,
                          batch_size=1, num_workers=1, patch_size=(1, 640, 400), queue_length=30, samples_per_volume=5)
    dm.prepare_data()
    dm.setup()
    print('1')
    for batch in dm.test_dataloader():
        print(batch.keys(), batch['name'])
        print(batch['image'].keys())
        print(batch['image']['data'].shape)
        img = batch['image']['data']

        img_dir = os.path.join('/home/Data/huangkun/Choroid/DATA/dataset4/test', batch['name'][0])
        os.makedirs(img_dir, exist_ok=True)
        img += 1
        img /= 2
        # img*= 255
        for j in range(img.shape[2]):
            img_path = os.path.join(img_dir, str(j + 1) + '.png')
            save_image(img[:, :, j, :, :], img_path)
        break
    print(len(dm.train_dataloader()))

    for batch in dm.train_dataloader():
        print(batch.keys(), batch['name'])
        img = batch['image']['data']
        label = batch['label']['data']
        # print(label)
        img +=1
        img /=2

        print(img.shape, batch['location'], str(batch['name'][0]))
        img_dir = os.path.join('/home/Data/huangkun/Choroid/DATA/dataset4/test', str(batch['name'][0])+'_2')
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir,str(batch['location'][0][0])+'.png')
        save_image(img[:, :, 0, :, :], img_path)
        img_dir2 = os.path.join('/home/Data/huangkun/Choroid/DATA/dataset4/test', str(batch['name'][0])+'_3')
        os.makedirs(img_dir2, exist_ok=True)
        label = label.float()
        img_path2 = os.path.join(img_dir2,str(batch['location'][0][0])+'.png')
        save_image(label[:, :, 0, :, :], img_path2)
        # exit()
