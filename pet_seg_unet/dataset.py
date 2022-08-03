# -----------------------------------------------------------------------#
#                          Library imports                              #
# -----------------------------------------------------------------------#
import glob
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchio as tio
import random
import nibabel as nib
import torch.nn.functional as f
import numpy as np


def NormalizeData(data):
    np.seterr(divide='ignore', invalid='ignore')
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class dataset_3D(Dataset):
    def __init__(self, partition, x, y, z, c, h, w, data_root_path, augment):
        self.partition = partition
        self.augment = augment
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.h = h
        self.w = w

        self.data_root_path = data_root_path

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        image_patches = []
        suv_patches = []
        mask_patches = []

        meshx, meshy, meshz = torch.meshgrid(
            (torch.linspace(-1, 1, self.x), torch.linspace(-1, 1, self.y), torch.linspace(-1, 1, self.z)))
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0)  # add batch dim
        data_path = glob.glob(self.data_root_path + '/*/*/')
        data_paths = [i for i in data_path if i[-6:-1] in self.partition]
        img = nib.load(data_paths[idx] + 'CTres.nii.gz')

        msk = nib.load(data_paths[idx] + 'SEG.nii.gz')
        mask = msk.get_fdata()
        mask = torch.as_tensor(mask)

        pet = nib.load(data_paths[idx] + 'SUV.nii.gz')

        image = img.get_fdata()
        image = NormalizeData(image)
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.type(torch.FloatTensor)

        mask = mask.permute(2, 0, 1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.type(torch.FloatTensor)

        suv = pet.get_fdata()
        suv = NormalizeData(suv)
        suv = torch.as_tensor(suv)
        suv = suv.permute(2, 0, 1)
        suv = suv.unsqueeze(0).unsqueeze(0)
        suv = suv.type(torch.FloatTensor)
        # create resized 3D CT and mask tensors
        # for the mask, to have either 0 or 1 as the voxel value, use 'nearest' for the interpoaltion mode
        image = f.grid_sample(image, grid, mode='bilinear', align_corners=True)
        suv = f.grid_sample(suv, grid, mode='bilinear', align_corners=True)
        mask = f.grid_sample(mask, grid, mode='nearest', align_corners=True)
        mask = mask.type(torch.IntTensor)
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        suv = suv.squeeze(0)
        image = image.permute(0, 3, 2, 1)
        mask = mask.permute(0, 3, 2, 1)
        suv = suv.permute(0, 3, 2, 1)
        if self.augment:
            aug = tio.Compose([tio.OneOf({tio.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 1, 1),
                                                           degrees=(5.0, 5.0, 0)): 0.35,
                                          tio.RandomElasticDeformation(num_control_points=9,
                                                                       max_displacement=(0.1, 0.1, 0.1),
                                                                       locked_borders=2,
                                                                       image_interpolation='linear'): 0.35,
                                          tio.RandomFlip(axes=(2,)): .3}),
                               ])
            subject = tio.Subject(ct=tio.ScalarImage(tensor=image),
                                  mask=tio.ScalarImage(tensor=mask),
                                  suv=tio.ScalarImage(tensor=suv))
            output = aug(subject)
            augmented_image = output['ct']
            augmented_mask = output['mask']
            augmented_suv = output['suv']

            image = augmented_image.data
            mask = augmented_mask.data
            suv = augmented_suv.data
        # for a [1, 1, 256, 256, 128] tensor:squeezing->[256,256,128]
        image = image.squeeze(0).squeeze(0)
        mask = mask.squeeze(0).squeeze(0)
        suv = suv.squeeze(0).squeeze(0)

        image = image.unfold(0, self.c, self.c)  # -->[4, 256, 128, 64]
        image = image.unfold(1, self.h, self.h)  # -->[4, 4, 128, 64, 64]
        image = image.unfold(2, self.w, self.w)  # -->[4, 4, 4, 64, 64, 32]

        mask = mask.unfold(0, self.c, self.c)
        mask = mask.unfold(1, self.h, self.h)
        mask = mask.unfold(2, self.w, self.w)

        suv = suv.unfold(0, self.c, self.c)
        suv = suv.unfold(1, self.h, self.h)
        suv = suv.unfold(2, self.w, self.w)

        image_patches.extend(image.contiguous().view(-1, self.c, self.h, self.w))
        mask_patches.extend(mask.contiguous().view(-1, self.c, self.h, self.w))
        suv_patches.extend(suv.contiguous().view(-1, self.c, self.h, self.w))
        image_patches = [i.unsqueeze(0) for i in image_patches]  # unsqueeze add 1
        mask_patches = [j.unsqueeze(0) for j in mask_patches]
        suv_patches = [k.unsqueeze(0) for k in suv_patches]

        return image_patches, mask_patches, suv_patches


class test_3D_dataset(Dataset):
    def __init__(self, CT_partition, pet_partition, augment=False):
        self.CT_partition = CT_partition
        self.pet_partition = pet_partition
        self.augment = augment

    def __len__(self):
        return len(self.CT_partition)

    def __getitem__(self, idx):
        # Generate one batch of data
        # ScalarImage expect 4DTensor, so add a singleton dimension
        image = self.CT_partition[idx].unsqueeze(0)
        pet = self.pet_partition[idx].unsqueeze(0)
        if self.augment:
            aug = tio.Compose([tio.OneOf \
                                   ({tio.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 1, 1),
                                                      degrees=(5.0, 5.0, 0)): 0.35,
                                     tio.RandomElasticDeformation(num_control_points=9,
                                                                  max_displacement=(0.1, 0.1, 0.1),
                                                                  locked_borders=2,
                                                                  image_interpolation='linear'): 0.35,
                                     tio.RandomFlip(axes=(2,)): .3}),
                               ])
            subject = tio.Subject(ct=tio.ScalarImage(tensor=image),
                                  pet=tio.ScalarImage(tensor=pet))
            output = aug(subject)
            augmented_image = output['ct']
            augmented_pet = output['pet']

            image = augmented_image.data
            pet = augmented_pet.data

        # note that mask is integer
        pet = pet.type(torch.FloatTensor)
        image = image.type(torch.FloatTensor)

        # The tensor we pass into ScalarImage is C x W x H x D, so permute axes to
        # C x D x H x W. At the end we have N x 1 x D x H x W.
        image = image.permute(0, 3, 2, 1)
        pet = pet.permute(0, 3, 2, 1)

        # Return image and mask pair tensors
        return image, pet
