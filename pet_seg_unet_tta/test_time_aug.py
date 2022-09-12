from email.header import make_header
import sys
from monai.utils import set_determinism
from monai.transforms.utils import allow_missing_keys_mode

from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord,
    Invertd,
    SaveImaged
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset, pad_list_data_collate
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


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.val_ds = None
        self.val_transforms = None
        self._model = UNet(
            dimensions=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=2, n_classes=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def prepare_data(self, data_dir):
        # set up the correct data path
        images_sg = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SEG*")))
        images_pt = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "CTres*")))

        data_dicts = [
            {"image_pt": image_name_pt, "image_ct": image_name_ct, "image_sg": image_name_sg}
            for image_name_pt, image_name_ct, image_name_sg in zip(images_pt[:4], images_ct[:4], images_sg[:4])
        ]
        val_files = data_dicts

        val_transforms = Compose(
            [
                LoadImaged(keys=["image_pt", "image_ct", "image_sg"]),
                AddChanneld(keys=["image_pt", "image_ct", "image_sg"]),

                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=-100, a_max=250,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                ScaleIntensityRanged(
                    keys=["image_pt"], a_min=0, a_max=15,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                # concatenate pet and ct channels
                ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
                # RandCropByPosNegLabeld(
                #     keys=["image_petct", "image_sg"],
                #     label_key="image_sg",
                #     spatial_size=(32, 32, 32),
                #     pos=1,
                #     neg=1,
                #     image_key="image_petct",
                #     image_threshold=1,
                # ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image_petct'],
                #                     mode=('bilinear'),
                #                     prob=1.0,
                #                     rotate_range=(0, 0, np.pi / 20),
                #                     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image_petct", "image_sg"]),

            ]
        )

        self.val_ds = Dataset(data=val_files, transform=val_transforms)
        return val_files

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate)
        return val_loader


def val_transform():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image_pt", "image_ct", "image_sg"]),
            AddChanneld(keys=["image_pt", "image_ct", "image_sg"]),

            # Spacingd(
            #  keys=["image_pt", "image_ct"],
            # pixdim=(2, 2, 3),
            # mode=("bilinear", "bilinear"),
            # ),
            # Orientationd(keys=["image_pt", "image_ct"], axcodes="LAS"),

            ScaleIntensityRanged(
                keys=["image_ct"], a_min=-100, a_max=250,
                b_min=0.0, b_max=1.0, clip=False,
            ),
            ScaleIntensityRanged(
                keys=["image_pt"], a_min=0, a_max=15,
                b_min=0.0, b_max=1.0, clip=False,
            ),
            ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

            RandCropByPosNegLabeld(
                keys=["image_petct", "image_sg"],
                label_key="image_sg",
                spatial_size=(32, 32, 32),
                pos=1,
                neg=1,
                image_key="image_petct",
                image_threshold=1,
            ),
            # RandAffined(
            #     keys=['image_petct'],
            #     mode=('bilinear'),
            #     prob=1.0,
            #     rotate_range=(0, 0, np.pi / 20),
            #     scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image_petct", "image_sg"]),
            ToTensord(keys=["image_petct", "image_sg"]),

        ]
    )
    return val_transforms


def segment_PETCT(ckpt_path, data_dir, export_dir):
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    net = Net.load_from_checkpoint(ckpt_path)
    net.eval()

    device = torch.device("cuda:0")
    net.to(device)
    net.prepare_data(data_dir)
    val_transforms = val_transform()
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image_ct",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="image_sg", to_onehot=2),
    ])
    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            # _img = pad_list_data_collate([val_data["image_petct"].to(device)])
            val_data["pred"] = sliding_window_inference(val_data["image_petct"].to(device), roi_size, sw_batch_size, net)
            seg = pad_list_data_collate([post_transforms(i) for i in val_data["pred"]])
            # transformed_data = val_transforms(val_files)
            seg.applied_operations = val_data["image_sg"].applied_operations
            seg_dict = {"image_sg": seg}
            with allow_missing_keys_mode(val_transforms):
                inverted_seg = val_transforms.inverse(seg_dict)
            # val_data["image_sg"] = val_data["image_sg"].to(device).squeeze()
            # val_data = [post_transforms(i) for i in decollate_batch(transformed_data)]
            # val_data["pred"] = [post_pred(j) for j in decollate_batch(val_data["pred"])]
            # val_data["image_sg"] = [post_label(k) for k in decollate_batch(val_data["image_sg"].to(device))]
            # val_outputs = post_pred(val_data["pred"])
            # val_labels = post_label(val_data["image_sg"].to(device))
            # SaveImaged(post_pred(val_data["pred"]))
            # val_outputs, val_labels = from_engine(["pred", "image_sg"])(val_data)
            # mean_dice = compute_meandice(include_background=False, y_pred=val_outputs, y=val_labels).cpu().numpy()
            # mean_dices.append(mean_dice[0][0])
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
    # avg = sum(mean_dices)/len(mean_dices)
    # print("mean dice for 100 data: ", avg)
            print("Metric on original image spacing: ", metric_org)


# def segment_PETdCT(ckpt_path, data_dir, export_dir):
#     print("starting")
#
#     # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
#     # post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2, n_classes=2)])
#     # post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2, n_classes=2)])
#     # post_pred = AsDiscrete(argmax=True, to_onehot=2, n_classes=2)
#     # post_label = AsDiscrete(to_onehot=2, n_classes=2)
#     net = Net.load_from_checkpoint(ckpt_path)
#     net.eval()
#
#     device = torch.device("cuda:0")
#     net.to(device)
#     net.prepare_data(data_dir)
#     mean_dices = []
#     with torch.no_grad():
#         for i, val_data in enumerate(net.val_dataloader()):
#             roi_size = (160, 160, 160)
#             sw_batch_size = 4
#
#             mask_out = sliding_window_inference(val_data["image_petct"].to(device), roi_size, sw_batch_size, net)
#             # mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
#             # mask_out = mask_out.astype(np.uint8)
#             # print("done inference")
#             #
#             # PT = nib.load(
#             #     os.path.join(data_dir, "SUV.nii.gz"))  # needs to be loaded to recover nifti header and export mask
#             # pet_affine = PT.affine
#             # PT = PT.get_fdata()
#             # mask_export = nib.Nifti1Image(mask_out, pet_affine)
#             # print(os.path.join(export_dir, "PRED.nii.gz"))
#             #
#             # nib.save(mask_export, os.path.join(export_dir, f"PRED{i}.nii.gz"))
#             # print("done writing")
#             # val_data["image_sg"] = val_data["image_sg"].to(device).squeeze()
#             # val_data["pred"] = [net.post_transform(i) for i in decollate_batch(val_data["pred"])]
#             # val_data["pred"] = [post_pred(j) for j in decollate_batch(val_data["pred"])]
#             # val_data["image_sg"] = [post_label(k) for k in decollate_batch(val_data["image_sg"].to(device))]
#             # val_outputs = post_pred(val_data["pred"])
#             # val_labels = post_label(val_data["image_sg"].to(device))
#             # SaveImaged(post_pred(val_data["pred"]))
#             # val_outputs, val_labels = from_engine(["pred", "image_sg"])(val_data)
#             # mean_dice = compute_meandice(include_background=False, y_pred=val_outputs, y=val_labels).cpu().numpy()
#             # mean_dices.append(mean_dice[0][0])
#             # compute metric for current iteration
#             # dice_metric = dice_metric(y_pred=val_outputs, y=val_labels)
#             # aggregate the final mean dice result
#         # metric_org = dice_metric.aggregate().item()
#         # reset the status for next validation round
#     # avg = sum(mean_dices)/len(mean_dices)
#     # print("mean dice for 100 data: ", avg)
#     # print("Metric on original image spacing: ", metric_org)
#

def run_inference(ckpt_path, data_dir, export_dir):
    segment_PETCT(ckpt_path, data_dir, export_dir)
#
#
# if __name__ == '__main__':
#     run_inference()
