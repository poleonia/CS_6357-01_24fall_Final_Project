import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
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
    RandRotate90d,
    ResizeWithPadOrCropd,
    SpatialCrop
)
from numpy import random
from monai.data import PatchDataset, DataLoader, PatchIter

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR,SwinUNETR,UNet
from torch.utils.tensorboard import SummaryWriter
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

from monai.networks.layers import Norm
import torch

img_files = []
seg_files = []
# for path in os.walk("/kaggle/input/kits19-1"):
for path in os.walk("/data2/HW/kits19/data/train"):
    for file in path[-1]:
        if 'imaging' in file:
            img_files.append(os.path.join(path[0],file))
        if 'segmentation' in file:
            seg_files.append(os.path.join(path[0],file))

val_img_files = []
val_seg_files = []
# for path in os.walk("/kaggle/input/kits19-1"):
for path in os.walk("/data2/HW/kits19/data/val"):
    for file in path[-1]:
        if 'imaging' in file:
            val_img_files.append(os.path.join(path[0],file))
        if 'segmentation' in file:
            val_seg_files.append(os.path.join(path[0],file))


#image list
img_files_for_training=[]
seg_files_for_training=[]
for file_index in range(len(img_files)):
    n1_img_sh = nib.load(img_files[file_index]).header.get_data_shape()
#     print(n1_img_sh)
    if     n1_img_sh[0]<500 :
        img_files_for_training.append(img_files[file_index])
        seg_files_for_training.append(seg_files[file_index])

img_files_for_val=[]
seg_files_for_val=[]
for file_index in range(len(val_img_files)):
    val_n1_img_sh = nib.load(val_img_files[file_index]).header.get_data_shape()
#     print(n1_img_sh)
    if     val_n1_img_sh[0]<500 :
        img_files_for_val.append(val_img_files[file_index])
        seg_files_for_val.append(val_seg_files[file_index])

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_files_for_training, seg_files_for_training)]
#train_files, val_files,test_files = data_dicts[:-15], data_dicts[-15:-7],data_dicts[-7:]
train_files = data_dicts

val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_files_for_val, seg_files_for_val)]



n1_img_sh = nib.load(data_dicts[0]['image'])#.header.get_data_shape()
data = n1_img_sh.get_fdata()


class ToKidney(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label[label==2]=1
        return {'image': image,
                'label': label}


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 32),  # 64
            pos=1,
            neg=1,
            num_samples=10,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,

        ),
        ResizeWithPadOrCropd(keys=["image", "label"],
                             spatial_size=(128, 128, 32)),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        )
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

train_ds =CacheDataset(
    data=train_files,
    transform=train_transforms,
)
train_loader = DataLoader(train_ds, batch_size=1, num_workers=8,shuffle=True)

val_ds =Dataset(
    data=val_files,
    transform=val_transforms,
)

from numpy import random
from monai.data import PatchDataset, DataLoader, PatchIter

inds = []


def Crop_label(label):
    tumor = 0
    if label.shape[0] == 1:
        label = label[0]
    x, y, z = label.shape
    while (tumor < 10):
        i = random.randint(x - 128)
        j = random.randint(y - 128)
        k = random.randint(z - 32)

        unique_num = np.unique(label[i:i + 128, j:j + 128, k:k + 32])
        if 2 in unique_num and tumor < 10:
            tumor = tumor + 1
            inds.append([i, j, k])

    return inds


def Crop_img(img):
    r = []


    if img.shape[0] == 1:
        img = img[0]

    for ind in inds:
        i, j, k = ind
        r.append(img[i:i + 128, j:j + 128, k:k + 32])
    return r

img = val_ds[0]["image"]
#print(img.shape)
label = val_ds[0]["label"]
n_samples = 10
inds=Crop_label(label)

ds_l = PatchDataset(data=label,
                  patch_func=Crop_img,
                  samples_per_image=n_samples)
#print(inds)
ds_i = PatchDataset(data=img,
                  patch_func=Crop_img,
                  samples_per_image=n_samples)
val_loader1=DataLoader(ds_i, batch_size=1, shuffle=False, num_workers=1)
val_loader2=DataLoader(ds_l, batch_size=1, shuffle=False, num_workers=1)


root_dir='/data2/HW/'
from monai.apps import load_from_mmar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model =UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels= (8,16,32,64),
    strides=(2,2,2),
).to(device)


loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4)

def validation(epoch_iterator_val):
    model.eval()
    val_labels_convert=[]
    val_outputs_convert=[]
    with torch.no_grad():
        for step,batch in enumerate(zip(val_loader1,val_loader2)):
            val_inputs,val_labels=batch
            val_inputs=val_inputs[None,:,:,:,:].cuda()
            val_labels=val_labels[ None,:, :,:,:].cuda()
            val_outputs = model(val_inputs)
            val_labels_convert.append(post_label(decollate_batch(val_labels)[0])[0])
            val_outputs_convert.append(post_pred(decollate_batch(val_outputs)[0])[0])
        dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    #model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader1, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
            else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
        global_step += 1
    return global_step, dice_val_best, global_step_best,epoch_loss_values

max_iterations = 100
eval_num = 50

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

model.load_state_dict(torch.load(os.path.join(root_dir, "last_model.pth")))
max_epochs = 200

for epoch in range(max_epochs):
    global_step, dice_val_best, global_step_best, epoch_loss_values = train(global_step, train_loader, dice_val_best,
                                                                            global_step_best)
####################################################################################################

########################################             Testing           ####################################
########################################################################################################
test_img_files = []
test_seg_files = []

for path in os.walk("/data2/HW/kits19/data/test"):
#for path in os.walk("/data2/HW/kits19/data/auto_test"):
    for file in path[-1]:
        if 'imaging' in file:
            test_img_files.append(os.path.join(path[0],file))
        if 'segmentation' in file:
            test_seg_files.append(os.path.join(path[0],file))

img_files_for_testing = []
seg_files_for_testing = []
for file_index in range(len(test_img_files)):
    n1_img_sh = nib.load(test_img_files[file_index]).header.get_data_shape()
    #print(file_index,":",n1_img_sh)
    # if n1_img_sh[0] < 500:
    if n1_img_sh[0] < 800:
        #print(file_index)
        img_files_for_testing.append(test_img_files[file_index])
        seg_files_for_testing.append(test_seg_files[file_index])

test_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_files_for_testing, seg_files_for_testing)]
test_files = test_data_dicts

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ]
)
test_ds =Dataset(
    data=test_files,
    transform=test_transforms,
)


def get_inds(img):
    inds = []
    if img.shape[0] == 1:
        img = img[0]
    x, y, z = img.shape
    for i in range(127, x, 128):
        for j in range(127, y, 128):
            for k in range(31, z, 32):
                inds.append([i, j, k])

    return inds

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.load_state_dict(torch.load(os.path.join(root_dir, "last_model.pth")))


from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation
import csv
output_path = 'test_result3/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

test_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS")
])

for tid, data in enumerate(test_ds):
    #print(f"Sample {i}: {data}")
    print(tid)
    if(tid<41):

        img = test_ds[tid]["image"].cuda()
        seg = test_ds[tid]['label'].cuda()
        indsf=get_inds(img)
        label=torch.tensor(np.zeros(img.shape)).cuda()
        for ind  in indsf:
            i,j,k=ind
            x=img[None,:,i-127:i+1,j-127:j+1,k-31:k+1]
            y = seg[None,:,i-127:i+1,j-127:j+1,k-31:k+1]
            label[0,i-127:i+1,j-127:j+1,k-31:k+1]=torch.argmax(model(x),axis=1)

        reference_nii = nib.load(test_data_dicts[tid]["label"])

        transformed_nii = test_transforms(test_data_dicts[tid]["label"])



        affine = reference_nii.affine
        header = reference_nii.header

        seg  = seg[0].permute(2,0,1)
        img  = img[0].permute(2,0,1)
        label  = label[0].permute(2,0,1)


        segmentation_nii = nib.Nifti1Image(label.cpu().numpy(), affine=affine)
        label_nii = nib.Nifti1Image(seg.cpu().numpy(), affine=affine)
        image_nii = nib.Nifti1Image(img.cpu().numpy(), affine=affine)




        result_path = os.path.join(output_path, test_data_dicts[tid]["image"].split('/')[-2])
        print(test_data_dicts[tid]["image"].split('/')[-2])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        nib.save(segmentation_nii, os.path.join(result_path,'preds.nii.gz'))
        nib.save(label_nii, os.path.join(result_path,'seg.nii.gz'))
        nib.save(image_nii, os.path.join(result_path,'imag.nii.gz'))