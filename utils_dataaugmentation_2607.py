
import os
import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage import rotate, gaussian_filter, shift, convolve
import matplotlib.pyplot as plt

def retrieve_image_array(path_to_image):
    image = sitk.ReadImage(path_to_image)
    array = sitk.GetArrayFromImage(image)
    # Retrieves pixel size in mm
    pixel_size = image.GetSpacing()

    return array, pixel_size


def retrieve_all_MHDimages(folder_mhd_images):
    """
    Funtion to get all the MHD images if we have previously saved them
    """
    images_mhd =[]
    for file in os.listdir(folder_mhd_images):
        if ".mhd" in file:
            image_name = file[:-4]
            path_to_image = os.path.join(folder_mhd_images,file)
            print ("Loading...",path_to_image)
            image_array, pixel_size = retrieve_image_array(path_to_image)
            images_mhd.append((image_name,image_array))
    return images_mhd, pixel_size




def get_image_and_calcium (data_path):
        for file in os.listdir(data_path):
                if "mask.mhd" in file:
                        mask_itk = sitk.ReadImage(os.path.join(data_path, file))
                        mask_array = sitk.GetArrayFromImage(mask_itk)
                elif file.startswith('r') and '.mhd' in file:
                        image_itk = sitk.ReadImage(os.path.join(data_path, file))
                        image_array = sitk.GetArrayFromImage(image_itk)
                
        return image_array, mask_array


def DA_flip (image, mask):

    if random.choice([True,False]):   
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if random.choice([True,False]):   
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    return image, mask
     

def DA_rotation (image, mask):
    angle = random.choice([20,10,5,355,350,340])
    rotated_image = rotate(image, angle, reshape = False)
    rotated_mask = rotate(mask, angle, reshape = False)
    return rotated_image, rotated_mask


def DA_translation (image, mask):
    shift_x = random.uniform(-30,30)
    shift_y = random. uniform(-30,30)

    # Nearest mode is chosen to fill the edges with the nearest pixel values
    shifted_image = shift(image, shift =[0,shift_x, shift_y], mode='nearest')
    shifted_mask = shift(mask, shift =[0,shift_x, shift_y], mode='nearest')
    
    return shifted_image, shifted_mask


def DA_noiseinjection(image, mask):
    mean = 0
    std = random.uniform(0, 0.1)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    
    return noisy_image, mask


def DA_blurring (image, mask):
     
     # Appl a Gaussian filter
     sigma = random.uniform(0, 2.0)
     blurred_image = gaussian_filter(image, sigma=sigma)

     return blurred_image, mask


def DA_sharpening (image, mask):
     
     kernel = np.array([[ 0, -1, 0], [-1, 5, -1], [0, -1, 0]])
     sharpened_image = np.zeros_like(image)
     for i in range(image.shape[0]):
          sharpened_image[i,:,:] = convolve(image[i,:,:], kernel)
     
     return sharpened_image, mask

def DA_edgedetection (image, mask):
     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
     edge_image = np.zeros_like(image)
     for i in range(image.shape[0]):
            edge_x = convolve(image[i,:, :], sobel_x)
            edge_y = convolve(image[i, :, :], sobel_y)
            edge_image[i, :, :] = np.sqrt(edge_x**2 + edge_y**2)

     return edge_image, mask


def DA_augment_choice(image, mask):
    augmentation_choice = random.choice(['geometric', 'noise', 'filter'])

    if augmentation_choice =='geometric':
        geometric_choice = random.choice(['flipping', 'rotation', 'translation'])
        if geometric_choice =='flipping':
              augmented_image, augmented_mask = DA_flip(image, mask)
        elif geometric_choice =='rotation':
              augmented_image, augmented_mask = DA_rotation(image, mask)
        elif geometric_choice =='translation':
              augmented_image, augmented_mask = DA_translation(image, mask) 
    
    elif augmentation_choice=="noise":
         augmented_image, augmented_mask = DA_noiseinjection(image, mask)

    elif augmentation_choice == "filter":
        filter_choice = random.choice(['smoothing', 'sharpening', 'edge detection'])
        if filter_choice == 'smoothing':
             augmented_image, augmented_mask = DA_blurring(image, mask)
        elif filter_choice == 'sharpening':
             augmented_image, augmented_mask = DA_sharpening(image, mask)
        if filter_choice == 'edge detection':
             augmented_image, augmented_mask = DA_edgedetection(image, mask)
    
    return augmented_image, augmented_mask


def save_augmenteddata (augmented_image_array, augmented_mask_array, data_name, data_folder):
     AG_image = sitk.GetImageFromArray(augmented_image_array)
     AG_mask = sitk.GetImageFromArray(augmented_mask_array)
     
     output_AG_folder = os.path.join(data_folder, 'AG_'+ data_name)
     if os.path.exists(output_AG_folder) is False:
        os.mkdir(output_AG_folder)

     output_path_AG_image = os.path.join(output_AG_folder, 'r_'+ data_name + '.mhd')
     output_path_mask = os.path.join(output_AG_folder, 'mask.mhd')
     sitk.WriteImage(AG_image, output_path_AG_image)
     sitk.WriteImage(AG_mask, output_path_mask)

    
# Do the choice of which data augmentation technique you want to apply
# Save augmented images
def plot_one_slice(image_name,image_array, slice_index ):
    """
    Function to plot a specific slice of the 3d images
    """
    # Extract the specific slice
    center_slice = image_array[slice_index,:,:]

    # Plot the center slice
    plt.imshow(center_slice, cmap='gray')
    plt.title(image_name)
    plt.axis('off')
    plt.show()