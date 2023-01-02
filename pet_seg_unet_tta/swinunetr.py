import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d, AddChanneld,
    EnsureTyped, ConcatItemsd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

print_config()

directory = "model_log"
root_dir = tempfile.mkdtemp() if directory is None else directory
print("root dir", root_dir)

num_samples = 4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = ["image_pt", "image_ct", "image_sg"]
train_transforms = Compose(
    [
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
        CropForegroundd(keys=keys, source_key="image_ct"),
        Orientationd(keys=keys, axcodes="LAS"),
        EnsureTyped(keys=keys, device=device, track_meta=True),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="image_sg",
            spatial_size=(96, 96, 96),
            pos=0,
            neg=1,
            num_samples=num_samples,
            image_key="image_ct",
            image_threshold=0,
        ),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image_petct", "image_sg"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image_petct", "image_sg"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image_petct"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
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
        CropForegroundd(keys=keys, source_key="image_ct"),
        Orientationd(keys=keys, axcodes="LAS"),
        ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),

        EnsureTyped(keys=["image_petct", "image_sg"], device=device, track_meta=True),
    ]
)

data_dir = '/home/navid/Desktop/Papers/MICCAI_challenge/registration -3D Unet/nifti/FDG-PET-CT-Lesions'
images_sg = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SEG*")))
images_pt = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "SUV*")))
images_ct = sorted(glob.glob(os.path.join(data_dir, 'PETCT*', '*', "CTres*")))

data_dicts = [
    {"image_pt": image_name_pt, "image_ct": image_name_ct, "image_sg": image_name_sg}
    for image_name_pt, image_name_ct, image_name_sg in zip(images_pt, images_ct, images_sg)
]
keys = ["image_pt", "image_ct", "image_sg"]
train_files, val_files = data_dicts[120:], data_dicts[:120]

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

# as explained in the "Setup transforms" section above, we want cached training images to not have metadata,
# and validations to have metadata the EnsureTyped transforms allow us to make this distinction on the other hand,
# set_track_meta is a global API; doing so here makes sure subsequent transforms (i.e., random transforms for
# training) will be carried out as Tensors, not MetaTensors
set_track_meta(True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=2,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# weight = torch.load("model_swinvit.pt")
# model.load_from(weights=weight)
# print("Using pretrained self-supervied Swin UNETR backbone weights !")

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image_petct"].cuda(), batch["image_sg"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 30000
eval_num = 500
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)

case_num = 4
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
# with torch.no_grad():
#     # img_name = os.path.split(val_ds[case_num]['image'].meta["filename_or_obj"])[1]
#     img = val_ds[case_num]["image_petct"]
#     label = val_ds[case_num]["image_sg"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(
#         val_inputs, (96, 96, 96), 4, model, overlap=0.8
#     )
