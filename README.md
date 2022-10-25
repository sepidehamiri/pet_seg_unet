# AutoPET Segmentation with Test-Time Augmentation

The code for the work "Improved automated lesion segmentation in whole-body FDG/PET-CT via Test-Time Augmentation
"(https://arxiv.org/abs/2210.07761).  we improve the network using a learnable composition of test time augmentations. We trained U-Net and Swin U-Netr on the training database to determine how different test time augmentation improved segmentation performance. We also developed an algorithm that finds an optimal test time augmentation contribution coefficient set. Using the newly trained U-Net and Swin U-Netr results, we defined an optimal set of coefficients for test-time augmentation and utilized them in combination with a pre-trained fixed nnU-Net. The ultimate idea is to improve performance at the time of testing when the model is fixed. Averaging the predictions with varying ratios on the augmented data can improve prediction accuracy. Our work has been accepted by AutoPET 2022 MICCAI challenge (https://autopet.grand-challenge.org/). I hope this will help you to reproduce the results.

