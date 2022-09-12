import os
import shutil
import tempfile
import glob
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d, EnsureType,
    ConcatItemsd, DivisiblePadd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
    pad_list_data_collate,
    Dataset
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0")
# torch.backends.cudnn.benchmark = True
print_config()

root_dir = "swinunetr_log"

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=2,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
            norm_name='batch'
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2, n_classes=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2, n_classes=2)])
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 800
        self.check_val = 30
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # prepare data
        data_dir = '/home/navid/Desktop/Papers/MICCAI_challenge/registration -3D Unet/nifti/FDG-PET-CT-Lesions'
        images_sg = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SEG*")))
        images_pt = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "CTres*")))

        data_dicts = [
            {"image_pt": image_name_pt, "image_ct": image_name_ct, "image_sg": image_name_sg}
            for image_name_pt, image_name_ct, image_name_sg in zip(images_pt[:5], images_ct[:5], images_sg[:5])
        ]
        keys = ["image_pt", "image_ct", "image_sg"]
        train_files, val_files = data_dicts[:4], data_dicts[4:]
        train_transforms = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(
                    keys=["image_pt", "image_ct"],
                    pixdim=(2, 2, 3),
                    mode=("bilinear", "bilinear"),
                ),
                Orientationd(keys=["image_pt", "image_ct"], axcodes="LAS"),
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=-100, a_max=250,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                ScaleIntensityRanged(
                    keys=["image_pt"], a_min=0, a_max=15,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                CropForegroundd(keys=keys, source_key="image_ct"),
                ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
                # RandCropByPosNegLabeld(
                #     keys=keys,
                #     label_key="image_sg",
                #     spatial_size=(128, 128, 128),
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image_petct",
                #     image_threshold=0,
                # ),
                DivisiblePadd(keys=["image_petct", "image_sg"], k=16),

                # concatenate pet and ct channels

            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(
                    keys=["image_pt", "image_ct"],
                    pixdim=(2, 2, 3),
                    mode=("bilinear", "bilinear"),
                ),
                Orientationd(keys=["image_pt", "image_ct"], axcodes="LAS"),
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=-100, a_max=250,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                ScaleIntensityRanged(
                    keys=["image_pt"], a_min=0, a_max=15,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                CropForegroundd(keys=keys, source_key="image_ct"),
                ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),
                DivisiblePadd(keys=["image_petct", "image_sg"], k=16),

            ]
        )

        self.train_ds = PersistentDataset(
            data=train_files, transform=train_transforms, cache_dir="cache_dir"
        )

        self.val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="cache_dir")

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=0, collate_fn=pad_list_data_collate)
        return val_loader

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=1, shuffle=True,
            num_workers=0, collate_fn=pad_list_data_collate,
        )
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image_petct"], batch["image_sg"])
        output = self.forward(images)
        output.set_data()
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image_petct"], batch["image_sg"]
        roi_size = (128, 128, 128)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}


def train_PETCT(log_dir='model_log'):
    # initialise the LightningModule
    net = Net()
    # net.prepare_data()
    # set up loggers and checkpoints
    log_dir = os.path.join(log_dir, "logs")
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir
    )

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=800,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=2,
    )
    torch.cuda.empty_cache()

    # train
    trainer.fit(net)
    print(
        f"train completed, best_metric: {net.best_val_dice:.4f} "
        f"at epoch {net.best_val_epoch}")


# checkpoint_callback = ModelCheckpoint(dirpath=root_dir, filename="best_metric_model")


if __name__ == '__main__':
    train_PETCT()
