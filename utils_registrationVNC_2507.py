from __future__ import print_function, absolute_import

import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import numpy as np
import imageio
import tkinter as tk
from scipy import ndimage




def crop_image_3d (image_3d, x_start, x_end, y_start, y_end, z_start, z_end, size):
    # Check dimensions of the 3D array
    # Ensure indices are within bounds
    x_start, x_end = max(0, x_start), min(size[0], x_end)
    y_start, y_end = max(0, y_start), min(size[1], y_end)
    z_start, z_end = max(0, z_start), min(size[2], z_end)
    
    # Crop the image using slicing
    cropped_image = image_3d[x_start:x_end, y_start:y_end, z_start:z_end]

    return cropped_image



def normalize_images (image,winsor_limits):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized =(image - min_val)/(max_val-min_val)
    # normalized = mstats.winsorize(normalized, winsor_limits)

    return normalized



def load_dicom_and_save_mhd(folder_path, output_mhd_file, winsor_limits):
    """
    Function that load dicom files and save them as mhd files after cropping them.
     MHD is a suitable format to be used in elastix
    E.g.: folder_path = r'D:\CT\FULLDOSE\REP1\0mms_VNCREP1'
    """
    
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    # Get the size of the image
    size = image.GetSize()
    
    # set limits to crop
    x_start=round(size[0]/2.5)
    x_end=size[0] - round(size[0]/2.5)
    y_start = round(size[1]/2.5)+25
    y_end = size[1] - round(size[1]/2.5)+25
    z_start = 0
    z_end = size[2]

    # Crop image
    cropped_image = crop_image_3d(image, x_start, x_end, y_start, y_end, z_start, z_end, size)

    # Save cropped image as mhd file in the assignated location    
    sitk.WriteImage(cropped_image, output_mhd_file)



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



def retrieve_image_array(path_to_image):
    image = sitk.ReadImage(path_to_image)
    array = sitk.GetArrayFromImage(image)
    # Retrieves pixel size in mm
    pixel_size = image.GetSpacing()

    return array, pixel_size




def detect_outliers(image, threshold):
    # Calculate the mean and standard deviation of pixel intensities in the image
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Compute the z-score for each pixel intensity
    z_scores = (image - mean_intensity) / std_intensity

    # Create a binary mask where outliers are labeled as True (1) and non-outliers as False (0)
    outlier_mask = np.abs(z_scores) > threshold

    # Return the binary mask indicating outlier pixels
    return outlier_mask




def initialise_elastix(path_to_elastix):
    ELASTIX_PATH = os.path.join(path_to_elastix, r'elastix.exe')
    TRANSFORMIX_PATH = os.path.join(path_to_elastix, r'transformix.exe')

    if not os.path.exists(ELASTIX_PATH):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(TRANSFORMIX_PATH):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    return ELASTIX_PATH, TRANSFORMIX_PATH




def retrieve_all_Dicomimages(VNCimages_folder_path, folder_mhd_images, winsorization_limit):
    """
    This function goes over all DICOM images in the given directory. It loads them and save them as mhd files
    in an automatic way.
    It also retrives an array with all the images arrays and image names
    """
    # Initialize a Tkinter root
    root = tk.Tk()
    root.withdraw()

    images_mhd =[]
    # Look for DICOM images in the directory
    for root_dir, dirs, files in os.walk(VNCimages_folder_path):
        if "VNC" in root_dir:
            print ("Loading...",root_dir)
            # The mhd image is going to be saved according to its standard/MCR category.
            # In otherwords, the last part of root_dir: if root_dir is 'D:\CT\FULLDOSE\REP1\0mms_VNCREP1'
            # then, filename will be 0mms_VNCREP1
            filename = os.path.basename(os.path.normpath(root_dir))
            image_mhd_path = os.path.join(folder_mhd_images,filename +".mhd")
            load_dicom_and_save_mhd(root_dir, image_mhd_path, winsorization_limit)

            # Read the just saved image and stored in the array
            mhd_image_array, pixel_size = retrieve_image_array(image_mhd_path)
            images_mhd.append((filename,mhd_image_array))

    return images_mhd, pixel_size
    



def calcification_detection(image_array, intensity_threshold, pixel_size, area_threshold=0.5):
    """
    Function to segment the calcifications in the "TNC" images.
    Done by using an intensity threshold of 130 and an area threshold of 0.5 mm2
    """

    # Calcification pixels are those above 130 HU
    thresholded_image = image_array > intensity_threshold

    # Connected Component Analysis
    labeled_array, num_features = ndimage.label(thresholded_image)
    # print(num_features)

    comp_image = np.zeros_like(thresholded_image)

    # Avoid errors in case no values above the intensity threshold are found
    if num_features !=0:
    
        for comp_label in range(1, num_features + 1):
            component  = (labeled_array == comp_label)

            # Since the threshold for the area is in 2D we have to study slice by slice
            for slice_index in range(component.shape[0]):
                component_size_slice = np.sum(component[slice_index,:,:])
                area = pixel_size[1]*pixel_size[2]*component_size_slice
            
                if area >= area_threshold:
                    comp_image[slice_index,:,:] = component[slice_index,:,:]         
    
    comp_image = comp_image.astype(np.uint8) * 255

    return comp_image


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


def plot_result_registration_simmetrics(results_folder, plot_title):
    """
    Function to plot the cost function during registration at different resolutions
    """
    # Plotting similarity metric
    iteration_files = [file for file in os.listdir(results_folder) if
                       file.startswith('IterationInfo')]
    fig = plt.figure()
    for file in iteration_files:
        log_path = os.path.join(results_folder, file)
        log = elastix.logfile(log_path)

        # Plot the 'metric' against the iteration number 'itnr'
        plt.plot(log['itnr'], log['metric'])
    plt.legend(['Resolution {}'.format(i) for i in range(len(iteration_files))])
    plt.title("Similarity metrics of registration " + plot_title)
    plt.show()




def plot_result_registration_images(results_folder_parameterfile, jacobian_image_array,fixed_image_array, moving_image_array, moving_image_name, mask_image_array, p_save_registered_folder, plot_title):
    """
    Funtion to visualize the resulting registered images after the alignment
    """

    # Select the slice index you want to visualize
    slice_index = fixed_image_array.shape[0]//2-50 

    # Retrieve the registered image
    registered_img_path=os.path.join(results_folder_parameterfile,'result.0.mhd')
    registeredt_image = sitk.ReadImage(registered_img_path)
    registered_image_array = sitk.GetArrayFromImage(registeredt_image)

    # Settings for the figure    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    slice_index = fixed_image_array.shape[0]//2 -50

    # The calcification mask has different colors according if it appears overlaied with the fixed (red), moving (green) or registered (blue) image
    mask_slice = mask_image_array[slice_index,:,:]
    mask_red = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
    mask_red[:, :, 0] = mask_slice
    mask_green = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
    mask_green[:, :, 1] = mask_slice
    mask_blue = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
    mask_blue[:, :, 2] = mask_slice


    ax.flatten()
    ax[0].imshow(fixed_image_array[slice_index, :, :], cmap='gray')
    ax[0].imshow(mask_red, alpha = 0.3, label = 'Calcium mask' )
    ax[0].axis('off')
    ax[0].set_title('Fixed image', fontsize=12)

    ax[1].imshow(moving_image_array[slice_index, :, :], cmap='gray')
    ax[1].imshow(mask_green, alpha = 0.3, label = 'Calcium mask' )
    ax[1].axis('off')
    ax[1].set_title('Moving image', fontsize=12)

    ax[2].imshow(registered_image_array[slice_index, :, :], cmap='gray')
    ax[2].imshow(mask_blue, alpha = 0.3, label = 'Calcium mask' )
    ax[2].axis('off')
    ax[2].set_title('Registered image', fontsize=12)
    
    im =ax[3].imshow(jacobian_image_array[slice_index, :, :])
    ax[3].axis('off')
    ax[3].set_title('Jacobian determinant', fontsize=12)
    fig.colorbar(im, ax=ax[3])

    fig.suptitle(plot_title)
    output_path_registered_image = os.path.join(p_save_registered_folder,moving_image_name+'.mhd')
    sitk.WriteImage(registeredt_image, output_path_registered_image)
    
    return registered_image_array




def align_images(moving_image_array, moving_image_name, fixed_image_array, fixed_image_name, mask_image_array, p_original_VNCimages_mhd, path_to_fixed_image, p_save_registered_folder, p_results_folder, p_parameters_folder, parameter_file, ELASTIX_PATH, TRANSFORMIX_PATH, plot_results_registration, plot_title):
    """
    Function to do the registration
    select the specific parameter file: according if you are doing rigid or non-rigid transformation
    Similarly, the folder p_original_VNCimages_mhd is modified to p_translated_images_mhd if we want to use the previous result of affine transformation for Bspline registration
    and the same occurs for p_translated_images_mhd that is substituted by p_final_images_mhd if we are in the same case
    """
    # Determine the oath to the parameter file
    path_to_parameter_file = os.path.join(p_parameters_folder,parameter_file)
    
    # Determine the path to the moving image
    path_to_moving_image = os.path.join(p_original_VNCimages_mhd, moving_image_name + '.mhd')
    print("Processing image :", moving_image_name)

    # Select where the results for this parameter file will be saved
    results_folder_parameterfile = os.path.join(p_results_folder, parameter_file)
    if os.path.exists(results_folder_parameterfile) is False:
        os.mkdir(results_folder_parameterfile)
    
    # Do registration
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    print("Fixed image:", path_to_fixed_image, "moving image: ", path_to_moving_image, "parameters path: ",
            path_to_parameter_file)
    el.register(
        fixed_image=path_to_fixed_image,
        moving_image=path_to_moving_image,
        parameters=[path_to_parameter_file],
        output_dir=results_folder_parameterfile)
    
    # Do transformation
    # Make a new transformix object tr with the CORRECT PATH to transformix
    transform_path = os.path.join(results_folder_parameterfile, 'TransformParameters.0.txt')
    tr = elastix.TransformixInterface(parameters=transform_path,
                                        transformix_path=TRANSFORMIX_PATH)
            
    # Get the Jacobian determinant
    jacobian_determinant_path = tr.jacobian_determinant(output_dir=p_results_folder)
    jacobian_image_array =imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))

    if plot_results_registration:
        plot_title1 = moving_image_name + " to " + fixed_image_name
        plot_title_general = plot_title + plot_title1
        plot_result_registration_simmetrics(results_folder=results_folder_parameterfile, plot_title=plot_title1)
        registered_image = plot_result_registration_images(results_folder_parameterfile, jacobian_image_array,fixed_image_array, moving_image_array, moving_image_name, mask_image_array, p_save_registered_folder, plot_title_general)
            
    return registered_image



def visualize_images(images_arrays, cols=5):
    """
    Function to visualize the images organized in fiver columns (this can be changed)
    """
    num_images = len(images_arrays)
    rows = (num_images // cols) + (1 if num_images % cols != 0 else 0) 

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  

    for i, ax in enumerate(axes):
        if i < num_images:
            image_array =  np.squeeze(images_arrays[i][1])  
            image_name = images_arrays[i][0] 
            slice_index =image_array.shape[0]//2 -50 
            ax.imshow(image_array[slice_index,:,:], cmap='gray')
            ax.set_title(image_name, fontsize=7)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()