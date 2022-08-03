# -----------------------------------------------------------------------#
#                          Library imports                              #
# -----------------------------------------------------------------------#
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict


def unet(n_epochs, loaders, model, optimizer, criterion, performance_metrics, path, threshold):
    # train 3D UNet for some number of epochs
    # keep track of loss and performance merics
    loss_and_metrics = []
    # initialize tracker for max DSC
    DSC_max = 0
    show_every = 100
    for epoch in tqdm(range(1, n_epochs + 1), total=n_epochs + 1):
        print(f'=== Epoch #{epoch} ===')
        # initialize variables to monitor training and validation loss, and performance metrics
        train_loss = 0.0
        valid_loss = 0.0

        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        valid_cnt = 0
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        for batch_idx, (images, targets, suvs) in enumerate(loaders['train']):
            # move to GPU
            # print(loaders['train'].dataset.partition[batch_idx])
            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            for image, target, suv in zip(images, targets, suvs):
                data = torch.cat((image, suv), 1)
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variable
                optimizer.zero_grad()
                # forward pass (inference)
                output = model(data)

                # calculate the batch loss

                # kernel = np.ones((3, 3), np.uint8)
                # t_edges = cv2.Canny(target, 0, 1)
                # t_dilation = ndimage.grey_dilation(t_edges, footprint=np.ones((3, 3))) / 255
                # t_dilation= cv2.dilate(t_edges, kernel, iterations=2) / 255

                # o_edges = cv2.Canny(output, 0, 1)
                # o_erosion = cv2.dilate(o_edges, kernel, iterations=2) / 255

                loss = criterion(output, target)
                # backpropagation
                loss.backward()
                # Update weights
                optimizer.step()
                # update training loss
                train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            ######################
        # validate the model #
        ######################
        print('=== Validation ===')
        # Set the model to inference mode
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, targets, suvs) in enumerate(loaders['valid']):
                # print(loaders['valid'].dataset.partition[batch_idx])
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["valid"])}...')
                for image, target, suv in zip(images, targets, suvs):
                    # move to GPU
                    data = torch.cat((image, suv), 1)
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    # forward pass (inference) to get the output
                    output = model(data)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # update validation loss
                    valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                    # convert output probabilities to predicted class
                    output = output.cpu().detach().numpy()
                    # Binarize the output
                    output_b = (output > threshold) * 1
                    output_b = np.squeeze(output_b)
                    batch_l = output_b.size
                    # update the total number of validation pairs
                    valid_cnt += batch_l
                    # Transform output back to Pytorch Tensor and move it to GPU
                    output_b = torch.as_tensor(output_b)
                    output_b = output_b.cuda()
                    # calculate average performance metrics per batches
                    m = performance_metrics(smooth=1e-6)
                    specificity, sensitivity, precision, F1_score, F2_score, DSC = m(output_b, target)
                    specificity_val += specificity * batch_l
                    sensitivity_val += sensitivity * batch_l
                    precision_val += precision * batch_l
                    F1_score_val += F1_score * batch_l
                    F2_score_val += F2_score * batch_l
                    DSC_val += DSC * batch_l
                    # Calculate the overall average metrics
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val / valid_cnt, sensitivity_val / valid_cnt, precision_val / valid_cnt, F1_score_val / valid_cnt, F2_score_val / valid_cnt, DSC_val / valid_cnt

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))
        print('Specificity: {:.6f} \tSensitivity: {:.6f} \tF2_score: {:.6f} \tDSC: {:.6f}'.format(
            specificity_val,
            sensitivity_val,
            F2_score_val,
            DSC_val
        ))

        if DSC_val > DSC_max and epoch > 5:
            print('Validation DSC increased.  Saving model ...')
            torch.save(model, path)
            torch.save(model, 'opt/algorithm/model.ckpt')
            DSC_max = DSC_val

        loss_and_metrics.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(),
                                 specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val))

    # save the loss_epoch as well as the performance metrics history
    df = pd.DataFrame.from_records(loss_and_metrics,
                                   columns=['epoch', 'Training_Loss', 'Validation_Loss', 'specificity', 'sensitivity',
                                            'precision', 'F1_score', 'F2_score', 'DSC'])
    df.to_csv('performance_metrics.csv', index=False)
    # get_ipython().system('cp performance_metrics.csv   output/performance_metrics.csv')
    torch.save(model, 'opt/algorithm/checkpoint.ckpt')
    # return trained model
    return model


class UNet_3D(nn.Module):
    # 3D UNet architecture
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p=0.5):
        super().__init__()
        features = init_features

        # Encoding layers
        self.encoder1 = UNet_3D._block(in_channels, features)
        self.encoder2 = UNet_3D._block(features, features * 2)
        self.encoder3 = UNet_3D._block(features * 2, features * 4)
        self.encoder4 = UNet_3D._block(features * 4, features * 8)

        # Bottleneck layer
        self.bottleneck = UNet_3D._block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_3D._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_3D._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_3D._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_3D._block(features * 2, features)

        # output layer
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        self.weight_init()

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Weight initialization
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    mean = 0
                    # standard deviation based on a 3*3 convolution
                    std = (2 / (3 * 3 * 3 * m.out_channels)) ** (0.5)
                    normal_init(m, mean, std)
            except:
                pass

    # Weight standardization:A normalization to be used with group normalization (micro_batch)
    def WS(self):
        for block in self._modules:
            if isinstance(block, nn.MaxPool2d) or isinstance(block, nn.ConvTranspose2d):
                pass
            else:
                for m in block:
                    if isinstance(m, nn.Conv2d):
                        # ref:https://github.com/joe-siyuan-qiao/WeightStandardization
                        weight = m.weight
                        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                                            keepdim=True).mean(dim=3, keepdim=True)
                        weight = weight - weight_mean
                        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                        weight = weight / std.expand_as(weight)
                        m.weight.data = weight

    def forward(self, x):
        # x = x.squeeze(0)
        # Encoding path
        enc1 = self.encoder1(x)
        p1 = self.dropout(self.pool(enc1))
        enc2 = self.encoder2(p1)
        p2 = self.dropout(self.pool(enc2))
        enc3 = self.encoder3(p2)
        p3 = self.dropout(self.pool(enc3))
        enc4 = self.encoder4(p3)
        p4 = self.dropout(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoding path
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.dropout(self.upconv3(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.dropout(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.dropout(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # self.WS()
        # Output
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_features, out_features):
        return nn.Sequential(OrderedDict([
            ("conv1", nn.Conv3d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
                bias=False)),
            ("norm1", nn.BatchNorm3d(num_features=out_features)),
            ("relu1", nn.ReLU(inplace=True)),
            # ("swish1", nn.SiLU(inplace=True)),
            ("conv2", nn.Conv3d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
                bias=False)),
            ("norm2", nn.BatchNorm3d(num_features=out_features)),
            ("relu2", nn.ReLU(inplace=True))
            # ("swish2", nn.SiLU(inplace=True))
        ]))
