from email.header import make_header
import sys
from monai.utils import set_determinism
from monai.transforms.utils import allow_missing_keys_mode
import time
from monai.transforms import (AsDiscrete, AsDiscreted, AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd, EnsureTyped, EnsureType,
                              ConcatItemsd, RandAffined, ToTensord, Invertd, SaveImaged, Lambdad, DivisiblePadd,
                              ScaleIntensityd, RandRotated, Resized, Rotated, ZoomD, GaussianSharpenD, GaussianSmoothD,
                              FlipD)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset, \
    pad_list_data_collate
from monai.handlers.utils import from_engine
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import pytorch_lightning
import time

import os
import glob
import numpy as np

import nibabel as nib

from monai_unet import Net

start_time = time.time()
device = torch.device("cuda:0")
data_dir = '/home/navid/Desktop/Papers/MICCAI_challenge/registration -3D Unet/nifti/FDG-PET-CT-Lesions'
ckpt_path = 'opt/algorithm/epoch=777-step=645733.ckpt'
images_sg = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SEG*")))
images_pt = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SUV*")))
images_ct = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "CTres*")))
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

val_files = [
    {"image_pt": image_name_pt, "image_ct": image_name_ct, "image_sg": image_name_sg}
    for image_name_pt, image_name_ct, image_name_sg in zip(images_pt[:5], images_ct[:5], images_sg[:5])
]
keys = ["image_pt", "image_ct", "image_sg"]

rotate_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct"]),
        # Lambdad("image_sg", lambda x: (x > 0).to(torch.float)),
        AddChanneld(keys=["image_pt", "image_ct"]),

        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ScaleIntensityRanged(
            keys=["image_pt"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        Rotated(angle=(np.pi / 18, np.pi / 18, np.pi / 18), keys=["image_pt", "image_ct"]),
        # CropForegroundd(keys, source_key="image_ct"),
        # DivisiblePadd(keys, 16),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ]
)

resize_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct", "image_sg"]),
        # Lambdad("image_sg", lambda x: (x > 0).to(torch.float)),
        AddChanneld(keys=["image_pt", "image_ct", "image_sg"]),

        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ScaleIntensityRanged(
            keys=["image_pt"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        Resized(keys=["image_pt", "image_ct", "image_sg"], spatial_size=(400, 400, 400),
                mode=["trilinear", "trilinear", "nearest"]),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ]
)

zoom_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct", "image_sg"]),
        # Lambdad("image_sg", lambda x: (x > 0).to(torch.float)),
        AddChanneld(keys=["image_pt", "image_ct", "image_sg"]),

        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ScaleIntensityRanged(
            keys=["image_pt"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ZoomD(keys=["image_pt", "image_ct", "image_sg"], zoom=0.9, mode=["trilinear", "trilinear", "nearest"]),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ]
)

gaussian_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct"]),
        # Lambdad("image_sg", lambda x: (x > 0).to(torch.float)),
        AddChanneld(keys=["image_pt", "image_ct"]),

        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ScaleIntensityRanged(
            keys=["image_pt"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        GaussianSharpenD(keys=["image_pt"]),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ]
)

blur_transforms = Compose(
    [
        LoadImaged(keys=["image_pt", "image_ct"]),
        # Lambdad("image_sg", lambda x: (x > 0).to(torch.float)),
        AddChanneld(keys=["image_pt", "image_ct"]),

        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        ScaleIntensityRanged(
            keys=["image_pt"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=False,
        ),
        GaussianSmoothD(keys=["image_ct"], sigma=0.6, approx="scalespace"),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ]
)

post_trans = Compose([
    AsDiscrete(argmax=True, to_onehot=2, n_classes=2),
])
post_label = AsDiscrete(to_onehot=2, n_classes=2)

original_transforms = Compose([
    LoadImaged(keys=keys),
    AddChanneld(keys=keys),

    ScaleIntensityRanged(
        keys=["image_ct"], a_min=-100, a_max=250,
        b_min=0.0, b_max=1.0, clip=False,
    ),
    ScaleIntensityRanged(
        keys=["image_pt"], a_min=0, a_max=15,
        b_min=0.0, b_max=1.0, clip=False,
    ),
    ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
    ToTensord(keys=["image_petct", "image_sg"]),
])

net = Net.load_from_checkpoint(ckpt_path)
net.eval()
net.to(device)


def infer_seg(images, model, roi_size=(160, 160, 160), sw_batch_size=2):
    val_outputs = sliding_window_inference(
        images, roi_size, sw_batch_size, model)
    return pad_list_data_collate([post_trans(i) for i in val_outputs])


g = 0.5
b = 0.5
u = 0.5
rotate = False
resize = False
zoom = False
gaussian = True
blur = True
orig_model = True
for file in val_files:
    # non transform data
    unmodified_data = original_transforms(file)
    unmodified_data["image_sg"] = post_label(unmodified_data["image_sg"])
    # _orig_img = pad_list_data_collate([unmodified_data["image_petct"].to(device)])
    # pred = infer_seg(_orig_img, net)[0].detach().cpu()
    # pred.applied_operations = unmodified_data["image_sg"].applied_operations
    # inverted_seg = {"image_sg": pred}

    if orig_model:
        transformed_data = original_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        # seg.applied_operations = transformed_data["image_sg"].applied_operations
        orig_model_seg = {"image_sg": seg}
        # with allow_missing_keys_mode(rotate_transforms):
        #     inverted_seg = rotate_transforms.inverse(seg_dict)
    # rotate data
    if rotate:
        transformed_data = rotate_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        seg.applied_operations = transformed_data["image_sg"].applied_operations
        seg_dict = {"image_sg": seg}
        with allow_missing_keys_mode(rotate_transforms):
            inverted_seg = rotate_transforms.inverse(seg_dict)
    if resize:
        transformed_data = resize_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        seg.applied_operations = transformed_data["image_ct"].applied_operations
        seg_dict = {"image_sg": seg}
        with allow_missing_keys_mode(resize_transforms):
            inverted_seg = resize_transforms.inverse(seg_dict)
    if zoom:
        transformed_data = zoom_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        seg.applied_operations = transformed_data["image_ct"].applied_operations
        seg_dict = {"image_sg": seg}
        with allow_missing_keys_mode(zoom_transforms):
            zoom_seg = zoom_transforms.inverse(seg_dict)
    if gaussian:
        transformed_data = gaussian_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        # seg.applied_operations = transformed_data["image_ct"].applied_operations
        gaussian_seg = {"image_sg": seg}
        # with allow_missing_keys_mode(gaussian_transforms):
        #     inverted_seg = gaussian_transforms.inverse(seg_dict)
    if blur:
        transformed_data = blur_transforms(file)
        _img = pad_list_data_collate([transformed_data["image_petct"].to(device)])
        seg = infer_seg(_img, net)[0].detach().cpu()
        # seg.applied_operations = transformed_data["image_sg"].applied_operations
        blur_seg = {"image_sg": seg}
        # with allow_missing_keys_mode(blur_transforms):
        #     inverted_seg = blur_transforms.inverse(seg_dict)
    seg_avg = (1 / (g + b + u)) * ((u * orig_model_seg["image_sg"]) +
                                   (g * gaussian_seg["image_sg"]) +
                                   (b * blur_seg["image_sg"]))
    seg_avg = post_trans(seg_avg)
    dice_metric(y_pred=seg_avg, y=unmodified_data["image_sg"])

metric_org = dice_metric.aggregate().item()
print("Metric on original image spacing: ", metric_org)
print("--- %s seconds ---" % (time.time() - start_time))
