import os
import glob
import torch
import random
import SimpleITK
import numpy as np
from net import unet
import nibabel as nib
import torch.utils.data
from net import UNet_3D
import torch.optim as optim
from scipy.ndimage import zoom
import torch.nn.functional as F
from metrics import TverskyLoss
from torchsummary import summary
from torch_lr_finder import LRFinder
from metrics import performance_metrics
from dataset import dataset_3D, test_3D_dataset


class Unet_patch_base:  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self, task='test'):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.task = task
        self.lr_find = False
        self.loaders = {}
        self.split_ratio = [0.5, 0.5, 0]  # A list of the (train,val,test) split ratio
        self.batch_size = 4
        self.num_workers = 0
        self.x = 256  # image x size
        self.y = 256  # image y size
        self.z = 128  # image z size
        self.h = 64  # patch height size
        self.w = 64  # patch width size
        self.c = 32  # patch channel size
        self.n_epochs = 100
        self.threshold = 0.5
        # Define type of optimizer as either 'Adam' or 'SGD'
        self.optimizer_type = 'Adam'
        self.data_root_path = '/nifti/FDG-PET-CT-Lesions'
        self.input_path = '/input/'
        # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'

        self.nii_path = '/opt/algorithm/'  # where to store the nii files
        self.ckpt_path = '/opt/algorithm/checkpoint.ckpt'
        # instantiate the unet
        self.model = UNet_3D(2, 1, 32, 0.2)
        # if GPU is available, move the model to GPU
        if torch.cuda.is_available():
            self.model.cuda()
        summary(self.model, (2, self.c, self.w, self.h), batch_size=self.batch_size)
        self.criterion = TverskyLoss(1e-8, 0.3, .7)
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=.0001)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        """
        If lr_find is True, after running this cell, assign the scheduler's max_lr to 
        the suggested maximum lr and then set lr_find to False in the "Set the parameters"
        section. Set the lr in the optimizer 1/10 of max_lr. Then re_run the code. 
        """

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def load_data(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        dataset = {}
        part = self.partitioning()
        if self.task == 'train':
            dataset['train'] = dataset_3D(
                part['train'], self.x, self.y, self.z, self.c, self.h, self.w, self.data_root_path, augment=True)
            self.loaders['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                num_workers=self.num_workers)
            dataset['valid'] = dataset_3D(
                part['valid'], self.x, self.y, self.z, self.c, self.h, self.w, self.data_root_path, augment=False)
            self.loaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                                batch_size=self.batch_size,
                                                                shuffle=False,
                                                                num_workers=self.num_workers)
            next(iter(self.loaders['train']))
            next(iter(self.loaders['valid']))
        else:
            CT_patches, pet_patches = self.patch_creator(part['test'], self.w, self.h, self.c, self.x, self.y, self.z)
            dataset_test = test_3D_dataset(CT_patches, pet_patches, augment=False)
            self.loaders['test'] = torch.utils.data.DataLoader(dataset_test,
                                                               batch_size=self.batch_size,
                                                               shuffle=False,
                                                               num_workers=0)
            # dataset['test'] = dataset_3D(
            #     part['test'], self.x, self.y, self.z, self.c, self.h, self.w, self.data_root_path, augment=False,
            #     test=True
            # )
            # self.loaders['test'] = torch.utils.data.DataLoader(dataset['test'],
            #                                                    batch_size=self.batch_size,
            #                                                    shuffle=False,
            #                                                    num_workers=self.num_workers)

        # batch = iter(self.loaders['valid'])
        # image, mask, pet = next(batch)
        # # for im, m, pt in zip(image[5], mask[5], pet[5]):
        #     # transfer C x D x H x W to C x W x H x D
        # m = mask[6][1:, :, :].squeeze()
        # im = image[6][1:, :, :].squeeze()
        # pt = pet[6][1:, :, :].squeeze()
        #
        # im = im.permute(2, 1, 0)
        # m = m.permute(2, 1, 0)
        # pt = pt.permute(2, 1, 0)
        #
        # im = im.numpy()
        # m = m.numpy()
        # pt = pt.numpy()
        #
        # plt.figure(figsize=(16, 16))
        # plt.subplot(4, 3, 1)
        # plt.imshow(im[:, :, 20], cmap="gray", interpolation=None)
        # plt.subplot(4, 3, 2)
        # plt.imshow(m[:, :, 20], cmap="gray", interpolation=None)
        # plt.subplot(4, 3, 3)
        # plt.imshow(pt[:, :, 20], cmap="gray", interpolation=None)
        # plt.imshow(m[:, :, 20], cmap="jet", alpha=0.3, interpolation=None)
        # plt.show()
        # elif self.task == 'test':
        #     ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        #     pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        #     self.loaders['test'] = os.path.splitext(ct_mha)[0]
        #
        #     self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
        #                             os.path.join(self.nii_path, 'SUV.nii.gz'))
        #     self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
        #                             os.path.join(self.nii_path, 'CTres.nii.gz'))
        return self.loaders

    def patch_creator(self, partition, kw, kh, kc, dw, dh, dc):
        # create 3D CT and mask patches (subvolumes)
        CT_patches = []
        pet_patches = []
        d1 = torch.linspace(-1, 1, dw)
        d2 = torch.linspace(-1, 1, dh)
        d3 = torch.linspace(-1, 1, dc)
        meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0)  # add batch dim
        for p in partition:
            CT_nifti = nib.load(p + 'CTres.nii.gz')
            pet_nifti = nib.load(p + 'SUV.nii.gz')
            ct_np = CT_nifti.get_fdata()
            pet_np = pet_nifti.get_fdata()
            ct = torch.tensor(ct_np)
            pet = torch.tensor(pet_np)
            ct = ct.permute(2, 0, 1)
            pet = pet.permute(2, 0, 1)
            ct = ct.unsqueeze(0).unsqueeze(0)
            pet = pet.unsqueeze(0).unsqueeze(0)
            ct = ct.type(torch.FloatTensor)
            pet = pet.type(torch.FloatTensor)
            CT_3D = F.grid_sample(ct, grid, mode='bilinear', align_corners=True)
            pet_3D = F.grid_sample(pet, grid, mode='bilinear', align_corners=True)
            ct = CT_3D.squeeze(0).squeeze(0)
            pet = pet_3D.squeeze(0).squeeze(0)
            # create subvolumes
            # it is like folding along width, then heigth, then depth
            # for a [1, 1, 256, 256, 128] tensor:squeezing->[256,256,128]
            CT_patch = ct.unfold(0, kw, kw)  # -->[4, 256, 128, 64]
            CT_patch = CT_patch.unfold(1, kh, kh)  # -->[4, 4, 128, 64, 64]
            CT_patch = CT_patch.unfold(2, kc, kc)  # -->[4, 4, 4, 64, 64, 32]

            pet_patch = pet.unfold(0, kw, kw)
            pet_patch = pet_patch.unfold(1, kh, kh)
            pet_patch = pet_patch.unfold(2, kc, kc)

            # add each patient's CT and mask subvolumes to their corresponding list
            CT_patches.extend(CT_patch.contiguous().view(-1, kw, kh, kc))
            pet_patches.extend(pet_patch.contiguous().view(-1, kw, kh, kc))

        return CT_patches, pet_patches

    def lr(self):
        """adjust the learning rate in the "Specify the loss function and optimizer" section"""
        """If you are willing to find the maximum learning rate using the One Cycle learning
         rate policy set lr_find to True"""
        if not self.lr_find:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.08,
                                                            steps_per_epoch=len(self.loaders['train']),
                                                            epochs=self.n_epochs)
            # (optimizer, max_lr=0.01, total_steps=4000)
        else:
            # https://github.com/davidtvs/pytorch-lr-finder
            desired_batch_size, real_batch_size = self.batch_size, self.batch_size
            accumulation_steps = desired_batch_size // real_batch_size
            lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device='cuda')
            lr_finder.range_test(self.loaders['train'], end_lr=1, num_iter=100, step_mode='exp')
            lr_finder.plot()  # to inspect the loss-learning rate graph
            lr_finder.reset()  # to reset the model and optimizer to their initial state

    def volume(self, num_patch_width, num_patch_height, num_patch_depth, num_batches,
               r, CT_subvol, predict_subvol, kc):
        image_vol = []
        prediction_vol = []
        # sweep in the depth direction
        for k in range(kc):
            idx = 0
            image = {}
            prediction = {}
            # sweep in the width and height direction to create layer k of the rth
            # subvolume horizontally stack the layer k of patches of each bach and
            # then sweep in the height direction and create an array for the kth
            # layer of the final 3D image. Then vertically stack all layers
            # to build a subvolume. Vertically stacking the subvolumes results
            # in a 3D image.
            for q in range(num_batches):
                for j, (im, pred) in enumerate(zip(CT_subvol[q], predict_subvol[q])):
                    if j % num_patch_depth == r:
                        # im = im[0:1, :, :, :]
                        im = np.squeeze(im).transpose(0, 2, 1)
                        pred = pred.transpose(0, 2, 1)
                        image[idx] = im[k, :, :]
                        prediction[idx] = pred[k, :, :]
                        idx += 1

            # for i in range(num_patch_width):
            #     for j in range(num_patch_height):
            #         hh = num_patch_width * i + j
            #         a = prediction[hh]
            #         predic = [a]
            #         vs = [np.hstack(tuple(predic))]
            #         blah = np.vstack(tuple(vs))
            #         prediction_vol.append(blah)
            image_vol.append(np.vstack(tuple([np.hstack(tuple([image[num_patch_width * i + j]
                                                               for j in range(num_patch_height)]))
                                              for i in range(num_patch_width)])))

            prediction_vol.append(np.vstack(tuple([np.hstack(tuple([prediction[num_patch_width * i + j]
                                                                    for j in range(num_patch_height)]))
                                                   for i in range(num_patch_width)])))

        return image_vol, prediction_vol

    def partitioning(self):
        patient_numbers = []
        if self.task == 'train':
            for patient in glob.glob(self.data_root_path + '/PETCT*/*'):
                patient_numbers.append(patient.replace(self.data_root_path + '/', "")[-5:])
        else:
            for patient in glob.glob(self.nii_path + '/CTres.nii.gz'):
                patient_numbers.append(patient[:-12])
        part = {'train': [], 'valid': [], 'test': []}
        # Shuffle the patient list
        random.shuffle(patient_numbers)
        length = len(patient_numbers)
        # find the split indices
        split_pt = [int(self.split_ratio[0] * length), int((self.split_ratio[0] + self.split_ratio[1]) * length)]
        # stratify split the paths
        part['train'] = patient_numbers[:split_pt[0]]
        part['valid'] = patient_numbers[split_pt[0]: split_pt[1]]
        part['test'] = [self.nii_path]
        print('train: ', len(part['train']), ' ', 'valid: ',
              len(part['valid']), ' ', 'test: ', len(part['test']), ' ',
              'total: ', len(part['train']) + len(part['valid']) + len(part['test']))
        # Return the partitions
        return part

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def train(self):
        self.model = unet(self.n_epochs, self.loaders, self.model, self.optimizer, self.criterion,
                          performance_metrics, self.ckpt_path, self.threshold)

    def predict(self):
        self.model = torch.load(self.ckpt_path)
        self.model.eval()
        predict_subvol = {}
        CT_subvol = {}

        for batch_idx, (data, pet) in enumerate(self.loaders['test']):
            # move to GPU
            data_cat = torch.cat((data, pet), 1)
            data_cat = data_cat.cuda()
            # forward pass
            output = self.model(data_cat)
            output = output.cpu().detach().numpy()
            # Binarize the output
            output_b = (output > self.threshold) * 1
            predict_subvol[batch_idx] = np.squeeze(output_b)
            CT_subvol[batch_idx] = np.squeeze(data.cpu().detach().numpy())
            # mask_subvol[batch_idx] = np.squeeze(target.cpu().detach().numpy())
        num_batches = 256 * 256 * 128 // (self.c * self.h * self.w * self.batch_size)
        num_patch_depth = 128 // self.c
        num_patch_width = 256 // self.w
        num_patch_height = 256 // self.h
        prediction_volume = []
        image_volume = []
        # sweep along the depth direction, create subvolumes and merge them to build
        # the final 3D image
        for r in range(num_patch_depth):
            image_vol, prediction_vol = self.volume(num_patch_width, num_patch_height,
                                                    num_patch_depth, num_batches,
                                                    r, CT_subvol,
                                                    predict_subvol, self.c)
            image_volume.extend(image_vol)
            prediction_volume.extend(prediction_vol)

        nifti_image_np = np.array(image_volume)
        nifti_prediction_np = np.array(prediction_volume).astype('int32')

        # resize and rotate
        nifti_image_np = np.transpose(nifti_image_np, (2, 1, 0))
        nifti_prediction_np = np.transpose(nifti_prediction_np, (2, 1, 0))

        img = nib.load(self.nii_path + 'CTres.nii.gz')
        img_affine = img.affine
        orig_size = img.shape

        nifti_image_np = zoom(nifti_image_np, (
            orig_size[0] / nifti_image_np.shape[0], orig_size[1] / nifti_image_np.shape[1],
            orig_size[2] / nifti_image_np.shape[2]))

        nifti_prediction_np = zoom(nifti_prediction_np, (
            orig_size[0] / nifti_prediction_np.shape[0], orig_size[1] / nifti_prediction_np.shape[1],
            orig_size[2] / nifti_prediction_np.shape[2]), mode='nearest')

        nifti_prediction = nib.Nifti1Image(nifti_prediction_np, affine=img_affine)  # Save axis for data (just identity)
        nifti_prediction.header.get_xyzt_units()
        nifti_prediction.to_filename(self.output_path + 'PRED.nii.gz')  # Save as NiBabel file

        nifti_image = nib.Nifti1Image(nifti_image_np, img_affine)  # Save axis for data (just identity)
        nifti_image.header.get_xyzt_units()
        nifti_image.to_filename(self.nii_path + 'image.nii.gz')  # Save as NiBabel file

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        uuid = self.load_inputs()
        self.load_data()
        if self.task == 'train':
            print('Start training')
            self.train()
        else:
            print('Start prediction')
            self.predict()

        # monai_unet.run_inference(self.ckpt_path, self.nii_path, self.output_path)
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    # task = train or test
    Unet_patch_base(task='test').process()
