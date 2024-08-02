
import numpy as np

import SimpleITK as sitk

import torchvision.transforms as transforms

import torch.nn as nn

import torch

import os

from scipy import ndimage

from matplotlib import pyplot as plt





def get_image_and_calcium (data_path):

        for file in os.listdir(data_path):

                if "mask.mhd" in file:

                        mask_itk = sitk.ReadImage(os.path.join(data_path, file))

                        mask_array = sitk.GetArrayFromImage(mask_itk)

                elif file.startswith('r') and '.mhd' in file:

                        image_itk = sitk.ReadImage(os.path.join(data_path, file))

                        image_array = sitk.GetArrayFromImage(image_itk)

                       

               

        return image_array, mask_array





class VNCDataset(torch.utils.data.Dataset):

    """

    Dataset containing VNC images.

    Parameters

    ----------

    paths : list[Path]

        paths to the patient data

    img_size : list[int]

        size of images to be interpolated to

    """

 

    def __init__(self, paths, img_size):

 

        # Initialize image and calcium mask list

        self.vnc_image_list = []

        self.camask_list = []

        self.scannames_list =[]

 

        # load images

        for path in paths:

            image_array, mask_array = get_image_and_calcium(path)

            self.vnc_image_list.append(image_array.astype(np.int32))

            self.camask_list.append(mask_array.astype(np.int32))

            self.scannames_list.append(os.path.basename(os.path.normpath(str(path))))

 

        # number of images and slices in the dataset

        self.no_scans = len(self.vnc_image_list)

        self.no_slices_perscan = self.vnc_image_list[0].shape[0]

 

        # Preprocessing:

        # transforms images --> convert PIL image to a PyTorch tensor

        self.img_transform = transforms.Compose(

            [

                transforms.ToPILImage(mode="I"),

                transforms.CenterCrop(img_size),

                transforms.ToTensor(),

            ]

        )

        # Standardise intensities based on mean and std deviation

        self.train_data_mean = np.mean(self.vnc_image_list)

        self.train_data_std = np.std(self.vnc_image_list)

        self.norm_transform = transforms.Normalize(

            self.train_data_mean, self.train_data_std

        )

 

    def __len__(self):

        """Returns length of dataset"""

        return self.no_scans * self.no_slices_perscan

 

    def __getitem__(self, index):

        """Returns the preprocessing VNC image and corresponding segementation

        for a given index.

 

        Parameters

        ----------

        index : int

            index of the image/segmentation in dataset

        """

 

        # compute which scan and which slice within that scan correspond to a given index

        # dataset is presented as long list of these slices

        vnc_scan = index // self.no_slices_perscan

        # calculate the slice number within the given scan

        the_slice = index - (vnc_scan * self.no_slices_perscan)

 

        image = self.vnc_image_list[vnc_scan][the_slice, ...]

        camask = (self.camask_list[vnc_scan][the_slice, ...] > 0).astype(np.int32)

        scan_name = self.scannames_list[vnc_scan]

 

        image = self.norm_transform(self.img_transform(image).float())

        camask = self.img_transform(camask)

 

        return (image, camask, scan_name)

 

        # return (

        #     self.norm_transform(

        #         self.img_transform(self.vnc_image_list[vnc_scan][the_slice, ...]).float()

        #     ),

        #     self.img_transform(

        #         (self.camask_list[vnc_scan][the_slice, ...] > 0).astype(np.int32)

        #     ),

        # )




class DiceBCELoss(nn.Module):

    """Loss function, computed as the sum of Dice score and binary cross-entropy.

 

    Notes

    -----

    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),

    and that the targets are integer values that represent the correct class labels.

    """

 

    def __init__(self):

        super(DiceBCELoss, self).__init__()

 

    def forward(self, outputs, targets, smooth=1):

        """Calculates segmentation loss for training

 

        Parameters

        ----------

        outputs : torch.Tensor

            predictions of segmentation model

        targets : torch.Tensor

            ground-truth labels

        smooth : float

            smooth parameter for dice score avoids division by zero, by default 1

 

        Returns

        -------

        float

            the sum of the dice loss and binary cross-entropy

        """

        outputs = torch.sigmoid(outputs)

 

        # flatten label and prediction tensors

        outputs = outputs.view(-1)

        targets = targets.view(-1)

 

        # compute Dice

        intersection = (outputs * targets).sum()

        dice_loss = 1 - (2.0 * intersection + smooth) / (

            outputs.sum() + targets.sum() + smooth

        )

        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

 

        return BCE + dice_loss




def dice_coefficient(output, target, smooth=1):

    """

    Compute the Dice coefficient between prediction and target.

    :param prediction: Predicted segmentation mask (tensor)

    :param target: Ground truth segmentation mask (tensor)

    :param smooth: Smoothing factor to avoid division by zero (default: 1e-6)

    :return: Dice coefficient

    """

    output = torch.sigmoid(output) # sqaush the output values bw 0 and 1, interpreting them as prob

 

    # Flatten label and pred tensors to 1D array (comp across all elements of tensors)

    output = output.view(-1)

    target = target.view(-1)

 

    intersection = (output * target).sum()

    dice = 1-(2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return dice





def plot_results (input, target, output, index, VNCscan_name):

       

    vnc_scan = index // 168

    # calculate the slice number within the given scan

    the_slice = index - (vnc_scan * 168)

 

    plt. figure(figsize=(15,5))

 

    plt.subplot(131)

    plt.imshow(input[0], cmap="gray")

    plt.title("Input")

 

    plt.subplot(132)

    plt.imshow(target[0])

    plt.title('Ground truth')

 

    plt.subplot(133)

    plt.imshow(output[0, 0])

    plt.title('Prediction')

 

    plt.suptitle(f"{VNCscan_name}: Slice: {the_slice} of scan: {vnc_scan}")

 

    plt.show()

 

    return the_slice






def crop_center(image_slice, IMAGE_SIZE):

    height, width = image_slice.shape

    center_x, center_y = width // 2, height // 2

   

    # Calculate the cropping coordinates

    start_x = center_x - IMAGE_SIZE[0] // 2

    end_x = center_x + IMAGE_SIZE[0] // 2

    start_y = center_y - IMAGE_SIZE[1] // 2

    end_y = center_y + IMAGE_SIZE[1] // 2

   

    cropped_slice = image_slice[start_y:end_y, start_x:end_x]

   

    return cropped_slice





def get_mhdInput (DATA_DIR, VNCscan_name, slice_no, IMAGE_SIZE):

    image_path = os.path.join (DATA_DIR, VNCscan_name)

    image_array, mask_array = get_image_and_calcium(image_path)

    image_slice = image_array[slice_no,:,:]

    cropped_image_slice = crop_center(image_slice, IMAGE_SIZE)

 

    return cropped_image_slice




def AgatsonScore_calculation(calcium_mask, DATA_DIR, VNCscan_name, slice_no, IMAGE_SIZE, pixel_size = 0.4296875):

 

    # the input slice should be in mhd format so we can relate to the real CT numbers in the image

    input = get_mhdInput(DATA_DIR, VNCscan_name, slice_no, IMAGE_SIZE)

    # Ensure input image is a numpy array (not a tensor)

    input = input.cpu().numpy() if torch.is_tensor(input) else input

    calcium_mask = calcium_mask.cpu().numpy() if torch.is_tensor(calcium_mask) else calcium_mask

 

    # CCA

    labeled_array, num_features = ndimage.label(calcium_mask)

    # print(num_features)

 

    total_AgatsonScore = 0

    for num_feature in range(1, num_features+1):

 

        # Extract the individual component

        component = (labeled_array == num_feature)

 

        # Compute the area

        numpixels_comp = np.sum(component)

        area = numpixels_comp*pixel_size*pixel_size

 

        # Find the maximum intensity of the calcification in the input image

        max_intensity = np.max(input[component])

 

        # Determine the weight correspondingly to the maximum intensity

        if max_intensity >=130:

            if max_intensity <200:

                weight = 1

            elif max_intensity < 300:

                weight = 2

            elif max_intensity < 400:

                weight = 3

            else:

                weight = 4

        else:

            weight = 0

       

        # Agatson score computation

        AgatsonScore = area*weight

        total_AgatsonScore += AgatsonScore

   

    # Interpretation of Agatson Score

    if total_AgatsonScore == 0:

        print("No detectable coronary artery calcification, which indicates a very low risk of coronary artery disease.")

    elif total_AgatsonScore >=1 and total_AgatsonScore <10:

        print("Minimal coronary artery clacification, which indicates a low risk of coronary artery disease.")

    elif total_AgatsonScore >=11 and total_AgatsonScore <100:

        print("Mild coronary artery clacification, which indicates moderate risk of coronary artery disease.")

    elif total_AgatsonScore >=101 and total_AgatsonScore <400:

        print("Moderate coronary artery clacification, which indicates a higher risk of coronary artery disease.")

    elif total_AgatsonScore >=400:

        print("Extensive coronary artery clacification, which indicates high risk of coronary artery disease.")

 

    return total_AgatsonScore

 