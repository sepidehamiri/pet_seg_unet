import sys

import SimpleITK
import time
import os

import subprocess
import shutil
import numpy as np
import nibabel as nib
from scipy import ndimage
# from nnunet.inference.predict import predict_from_folder
from predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch


class Autopet_baseline:  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        self.nii_seg_file = 'TCIA_001.nii.gz'
        self.nii_seg_blur_ct = 'TCIA_002.nii.gz'
        self.nii_seg_sharp_pet = 'TCIA_003.nii.gz'
        self.nii_seg_blur = 'TCIA_004.nii.gz'
        self.nii_seg_sharp = 'TCIA_005.nii.gz'
        self.nii_seg_blurct_sharppet = 'TCIA_006.nii.gz'

        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
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

    def load_inputs(self, tta):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))

        if tta == 'blur_ct':
            pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
            ct = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
            pet_affine = pet.affine

            pet_a = np.array(pet.dataobj)
            ct_a = np.array(ct.dataobj)

            ct_a = ndimage.gaussian_filter(ct_a, sigma=0.6)
            # ct_a = gaussian_filter(ct_a, sigma=1)
            pet_nib = nib.Nifti1Image(pet_a, pet_affine)
            ct_nib = nib.Nifti1Image(ct_a, pet_affine)

            nib.save(pet_nib, os.path.join(self.nii_path, 'TCIA_002_0000.nii.gz'))
            nib.save(ct_nib, os.path.join(self.nii_path, 'TCIA_002_0001.nii.gz'))
            print('Save Blur PET!')

        if tta == 'sharp_pet':
            pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
            ct = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
            pet_affine = ct.affine

            ct_a = np.array(ct.dataobj)
            pt_a = np.array(pet.dataobj)
            blurred_f = ndimage.gaussian_filter(pt_a, 3)
            filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
            alpha = 30
            sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

            pet_nib = nib.Nifti1Image(sharpened, pet_affine)
            ct_nib = nib.Nifti1Image(ct_a, pet_affine)

            nib.save(pet_nib, os.path.join(self.nii_path, 'TCIA_003_0000.nii.gz'))
            nib.save(ct_nib, os.path.join(self.nii_path, 'TCIA_003_0001.nii.gz'))
            print('Save sharp CT')
        if tta == 'blur':
            pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
            ct = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
            pet_affine = ct.affine

            pet_a = np.array(pet.dataobj)
            ct_a = np.array(ct.dataobj)

            pet_a = ndimage.gaussian_filter(pet_a, sigma=0.3)
            ct_a = ndimage.gaussian_filter(ct_a, sigma=0.6)
            pet_nib = nib.Nifti1Image(pet_a, pet_affine)
            ct_nib = nib.Nifti1Image(ct_a, pet_affine)

            nib.save(pet_nib, os.path.join(self.nii_path, 'TCIA_004_0000.nii.gz'))
            nib.save(ct_nib, os.path.join(self.nii_path, 'TCIA_004_0001.nii.gz'))
            print('Sve zoom in CT and PET!')
        if tta == 'sharp':
            pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
            ct = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
            pet_affine = ct.affine

            pt_a = np.array(pet.dataobj)
            ct_a = np.array(ct.dataobj)
            blurred_pet = ndimage.gaussian_filter(pt_a, 3)
            filter_blurred_pet = ndimage.gaussian_filter(blurred_pet, 1)
            alpha = 30
            sharpened_pet = blurred_pet + alpha * (blurred_pet - filter_blurred_pet)

            blurred_ct = ndimage.gaussian_filter(ct_a, 3)
            filter_blurred_ct = ndimage.gaussian_filter(blurred_ct, 1)
            alpha = 30
            sharpened_ct = blurred_ct + alpha * (blurred_ct - filter_blurred_ct)
            pet_nib = nib.Nifti1Image(sharpened_pet, pet_affine)
            ct_nib = nib.Nifti1Image(sharpened_ct, pet_affine)

            nib.save(pet_nib, os.path.join(self.nii_path, 'TCIA_005_0000.nii.gz'))
            nib.save(ct_nib, os.path.join(self.nii_path, 'TCIA_005_0001.nii.gz'))
            print('Save zoom out PET CT!')
        if tta == 'blurct_sharppet':
            pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
            ct = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
            pet_affine = ct.affine

            pet_a = np.array(pet.dataobj)
            ct_a = np.array(ct.dataobj)

            blurred_f = ndimage.gaussian_filter(pet_a, 3)
            filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
            alpha = 30
            sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
            ct_a = ndimage.gaussian_filter(ct_a, sigma=0.9)

            pet_nib = nib.Nifti1Image(sharpened, pet_affine)
            ct_nib = nib.Nifti1Image(ct_a, pet_affine)

            nib.save(pet_nib, os.path.join(self.nii_path, 'TCIA_006_0000.nii.gz'))
            nib.save(ct_nib, os.path.join(self.nii_path, 'TCIA_006_0001.nii.gz'))
            print('Save zoom out PET CT!')
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        o = 0.6
        # sp = 0.1
        bc = 0.5
        # b = 0.5
        # s = 0.1
        # bs = 0.1

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        orig = nib.load(os.path.join(self.result_path, self.nii_seg_file))
        # blur = nib.load(os.path.join(self.result_path, self.nii_seg_blur))
        # sharp = nib.load(os.path.join(self.result_path, self.nii_seg_sharp))
        blur_ct = nib.load(os.path.join(self.result_path, self.nii_seg_blur_ct))
        # sharp_pet = nib.load(os.path.join(self.result_path, self.nii_seg_sharp_pet))
        # bc_sp = nib.load(os.path.join(self.result_path, self.nii_seg_blurct_sharppet))
        orig_affine = orig.affine
        orig_a = np.array(orig.dataobj)
        # blur_a = np.array(blur.dataobj)
        # sharp_a = np.array(sharp.dataobj)
        blur_ct_a = np.array(blur_ct.dataobj)
        # sharp_pet_a = np.array(sharp_pet.dataobj)
        # bc_sp_a = np.array(bc_sp.dataobj)
        n = o + bc
        # seg_avg = (1 / n) * (bs * bc_sp_a)

        seg_avg = (1 / n) * ((o * orig_a) + (bc * blur_ct_a))
        print('SEG AVG shape', seg_avg.shape)
        seg_avg[seg_avg > 0.5] = 1
        seg_avg[seg_avg < 0.5] = 0

        avg_seg_nib = nib.Nifti1Image(seg_avg, orig_affine)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        nib.save(avg_seg_nib, os.path.join(self.result_path, self.nii_seg_file))
        self.convert_nii_to_mha(os.path.join(self.result_path, self.nii_seg_file),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        # cproc = subprocess.run(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres',
        # shell=True, check=True) os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m
        # 3d_fullres')
        print("nnUNet segmentation starting!")
        input_folder = self.nii_path
        output_folder = self.result_path
        part_id = 0  # args.part_id
        num_parts = 1  # args.num_parts
        folds = 'None'  # args.folds
        save_npz = False  # args.save_npz
        lowres_segmentations = 'None'  # args.lowres_segmentations
        num_threads_preprocessing = 1  # args.num_threads_preprocessing
        num_threads_nifti_save = 1  # args.num_threads_nifti_save
        disable_tta = False  # args.disable_tta
        step_size = 0.5  # args.step_size
        # interp_order = args.interp_order
        # interp_order_z = args.interp_order_z
        # force_separate_z = args.force_separate_z
        overwrite_existing = False  # args.overwrite_existing
        mode = 'normal'  # args.mode
        all_in_gpu = 'None'  # args.all_in_gpu
        model = '3d_fullres'  # args.model
        trainer_class_name = default_trainer  # args.trainer_class_name
        cascade_trainer_class_name = default_cascade_trainer  # args.cascade_trainer_class_name
        disable_mixed_precision = False  # args.disable_mixed_precision
        plans_identifier = default_plans_identifier
        chk = 'model_final_checkpoint'

        task_name = '001'

        if not task_name.startswith("Task"):
            task_id = int(task_name)
            task_name = convert_id_to_task_name(task_id)

        assert model in ["2d", "3d_lowres", "3d_fullres",
                         "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                "3d_cascade_fullres"

        if lowres_segmentations == "None":
            lowres_segmentations = None

        if isinstance(folds, list):
            if folds[0] == 'all' and len(folds) == 1:
                pass
            else:
                folds = [int(i) for i in folds]
        elif folds == "None":
            folds = None
        else:
            raise ValueError("Unexpected value for argument folds")

        assert all_in_gpu in ['None', 'False', 'True']
        if all_in_gpu == "None":
            all_in_gpu = None
        elif all_in_gpu == "True":
            all_in_gpu = True
        elif all_in_gpu == "False":
            all_in_gpu = False

        # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
        # In that case we need to try and predict with 3d low res first
        if model == "3d_cascade_fullres" and lowres_segmentations is None:
            print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
            assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
                                                    "inference of the cascade, custom values for part_id and num_parts " \
                                                    "are not supported. If you wish to have multiple parts, please " \
                                                    "run the 3d_lowres inference first (separately)"
            model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
                                     plans_identifier)
            assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
            lowres_output_folder = join(output_folder, "3d_lowres_predictions")
            predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
                                num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts,
                                not disable_tta,
                                overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                                mixed_precision=not disable_mixed_precision,
                                step_size=step_size)
            lowres_segmentations = lowres_output_folder
            torch.cuda.empty_cache()
            print("3d_lowres done")

        if model == "3d_cascade_fullres":
            trainer = cascade_trainer_class_name
        else:
            trainer = trainer_class_name

        model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                                 plans_identifier)
        print("using model stored in ", model_folder_name)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

        predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

        print("nnUNet segmentation done!")
        if not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('waiting for nnUNet segmentation to be created')
        while not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('.', end='')
            time.sleep(5)
        # print(cproc)  # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code
        # is received but segmentation file not yet written. This hack ensures that all spawned subprocesses are
        # finished before being printed.
        print('Prediction finished')
        print("nnUNet segmentation done!")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print('Start processing')
        tta = ['original', 'blur_ct']
        # tta = ['original', 'blur', 'sharp', 'blur_ct', 'sharp_pet', 'blurct_sharppet']
        for i in tta:
            uuid = self.load_inputs(i)
            print('Start prediction')
            self.predict()
            print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
