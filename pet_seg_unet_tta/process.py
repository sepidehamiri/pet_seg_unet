import SimpleITK
import numpy as np
import torch

import monai_unet
import os
import shutil


class Unet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # self.input_path = 'input/'  # according to the specified grand-challenge interfaces
        self.output_path = 'output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        # self.nii_path = 'opt/algorithm/'  # where to store the nii files
        self.nii_path = '/home/navid/Desktop/Papers/MICCAI_challenge/registration -3D Unet/nifti/FDG-PET-CT-Lesions'
        # self.nii_path = '/home/navid/Desktop/Papers/MICCAI_challenge/registration -3D Unet/nifti/FDG-PET-CT-Lesions/PETCT_0168f65af8/04-04-2003-NA-PET-CT Ganzkoerper  primaer mit KM-82224'
        self.ckpt_path = 'opt/algorithm/epoch=777-step=64573.ckpt'

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

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

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self, inputs):
        """
        Your algorithm goes here
        """
        pass
        # return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        # uuid = self.load_inputs()
        print('Start prediction')
        monai_unet.run_inference(self.ckpt_path, self.nii_path, self.output_path)
        print('Start output writing')
        # self.write_outputs(uuid)


if __name__ == "__main__":
    Unet_baseline().process()
