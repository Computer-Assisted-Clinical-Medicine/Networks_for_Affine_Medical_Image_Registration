import os
import numpy as np
import pandas as pd
import config as cfg
import SimpleITK as sitk
from processing import seg_binary
import image


def load_image(path,filename):
    '''
    Loads Nifti images and returns a numpy array.

    @param path: The path to the Nifti file
    @param filename: The name of the Nifti file
    @return: A numpy array containing the pixel data of the Nifti file
    '''
    # Use a SimpleITK reader to load the nii images and labels for training
    data_img = sitk.ReadImage(os.path.join(path, filename), sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(data_img)
    cfg.data_background_value=int(np.min(data))

    #Resample
    data_img= resample(data_img)
    data = sitk.GetArrayFromImage(data_img)

    # Normalize, determine the data type based on the path
    if path ==cfg.path_fixed:
        data = normalize(data, data_type="fixed")
    elif path==cfg.path_moving:
        data = normalize(data, data_type="moving")
    elif path==cfg.path_seg_fixed or path==cfg.path_seg_moving:
        data = normalize(data, data_type="label")

    # move z axis to last index
    data = np.moveaxis(data, 0, -1)
    if cfg.print_details:
        print(data.shape)

    return data

def load_image_data_info_fixed_image(path,filename):
    '''
    Loads Nifti images and returns a numpy array.
    Uses information of fixed image to resample the new image

    @param path: The path to the Nifti file
    @param filename: The name of the Nifti file
    @return: A numpy array containing the pixel data of the Nifti file
    '''
    # Use a SimpleITK reader to load the nii images and labels for training
    data_img = sitk.ReadImage(os.path.join(path, filename))

    data = sitk.GetArrayFromImage(data_img)
    cfg.data_background_value=int(np.min(data))

    orig_img = sitk.ReadImage(cfg.orig_filepath)
    data_info = image.get_data_info(orig_img)

    # Resample
    data_img= resample_data_info_fixed_image(data_img, data_info)

    data = sitk.GetArrayFromImage(data_img)

    # move z axis to last index
    data = np.moveaxis(data, 0, -1)
    return data


def save_image(data,path,filename, use_orig_file_info=False, do_resample=True):
    """
        Saves a 3D image in ITK format (.nii.gz) to disk.

        Args:
            data (numpy.ndarray): A 3D numpy array containing the image data.
            path (str): The directory path where the image should be saved.
            filename (str): The filename of the image (without file extension).
            use_orig_file_info (bool): If True, uses the metadata from the original image file
                                       to save the new image (default False).
            do_resample (bool): If True, resamples the image to the target resolution and size
                                specified in the config file (default True).

        Returns:
            None
        """
    # Set data info based on whether to use original file info or config file info
    if use_orig_file_info:
        orig_img = sitk.ReadImage(cfg.orig_filepath)
        data_info = image.get_data_info(orig_img)
    else:
        data_info = {}
        if cfg.adapt_resolution:
            data_info['target_spacing'] = cfg.target_spacing
            data_info['target_size'] = cfg.target_size
            data_info['target_type_image'] = cfg.target_type_image

    # Set remaining data info fields
    data_info['res_spacing'] = cfg.target_spacing
    data_info['res_origin'] = [0.0, 0.0, 0.0]
    data_info['res_direction'] = cfg.target_direction

    # Move axis of data array to match ITK format
    data = np.moveaxis(data, -1, 0)

    # Convert numpy array to ITK image object and save to disk
    data_out = image.np_array_to_itk_image(data, data_info, do_resample=do_resample,
                                           out_type=sitk.sitkVectorFloat32,
                                           background_value=cfg.label_background_value,
                                           interpolator=sitk.sitkLinear)
    sitk.WriteImage(data_out, os.path.join(path,filename))

def resample(data):
    '''!
    This function operates as follows:
    - extract image meta information
    - augmentation is only on in training
    - calls the static function _resample()

    @param data <em>ITK image,  </em> patient image
    @return resampled data and label images
    '''

    target_info = {}
    target_info['target_spacing'] = cfg.target_spacing
    target_info['target_direction'] = cfg.target_direction
    target_info['target_size'] = cfg.target_size
    target_info['target_type_image'] = cfg.target_type_image
    target_info['target_type_label'] = cfg.target_type_image

    do_augment = False
    cfg.max_rotation=0

    data.SetDirection(cfg.target_direction)

    return image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                     do_adapt_resolution=cfg.adapt_resolution,
                                     do_augment=do_augment,
                                     max_rotation_augment=cfg.max_rotation)

def resample_data_info_fixed_image(data, data_info):
    '''!
    This function operates as follows:
    - extract image meta information
    - augmentation is only on in training
    - calls the static function _resample()

    @param data <em>ITK image,  </em> patient image
    @return resampled data and label images
    '''

    target_info = {}
    target_info['target_spacing'] = data_info['orig_spacing']
    target_info['target_direction'] = data_info['orig_direction']
    target_info['target_size'] = data_info['orig_size']
    target_info['target_type_image'] = cfg.target_type_image
    target_info['target_type_label'] = cfg.target_type_image

    do_augment = False
    cfg.max_rotation=0

    data.SetDirection(cfg.target_direction)

    img= image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                     do_adapt_resolution=cfg.adapt_resolution,
                                     do_augment=do_augment,
                                     max_rotation_augment=cfg.max_rotation)

    if np.min(sitk.GetArrayFromImage(img))==np.max(sitk.GetArrayFromImage(img)): #resample didn't work correctly
        data.SetDirection(data_info['orig_direction'])

        img = image.resample_sitk_image(data, target_info, data_background_value=cfg.data_background_value,
                                          do_adapt_resolution=cfg.adapt_resolution,
                                          do_augment=do_augment,
                                          max_rotation_augment=cfg.max_rotation)

    return img

def normalize(img, eps=np.finfo(np.float).min, data_type="fixed"):
    '''
    Truncates input to interval [config.norm_min_v, config.norm_max_v] and
     normalizes it to interval [-1, 1] when using WINDOW and to mean = 0 and std = 1 when MEAN_STD.
    '''
    if data_type == "label":
        img=seg_binary(img)

    elif cfg.normalizing_method == cfg.NORMALIZING.WINDOW:
        if data_type=="fixed":
            cfg.norm_min_v = cfg.norm_min_v_fixed
            cfg.norm_max_v = cfg.norm_max_v_fixed
        elif data_type=="moving":
            cfg.norm_min_v = cfg.norm_min_v_moving
            cfg.norm_max_v = cfg.norm_max_v_moving

        flags = img < cfg.norm_min_v
        img[flags] = cfg.norm_min_v
        flags = img > cfg.norm_max_v
        img[flags] = cfg.norm_max_v
        img = (img - cfg.norm_min_v) / (cfg.norm_max_v - cfg.norm_min_v + cfg.norm_eps)
        img = img * cfg.intervall_max # interval [0, 1]

    elif cfg.normalizing_method == cfg.NORMALIZING.MEAN_STD:
        img = img - np.mean(img)
        std = np.std(img)
        img = img / (std if std != 0 else eps)

    if cfg.print_details:
        print("img min and max:",img.min(), img.max())

    return img

def getdatalist_from_csv(fixed_csv, moving_csv):
    """
        Reads two csv files containing the paths of fixed and moving images, and returns two lists of paths respectively.

        Args:
            fixed_csv (str): path to the csv file containing the paths of fixed images
            moving_csv (str): path to the csv file containing the paths of moving images

        Returns:
            tuple: a tuple containing two lists of paths of fixed and moving images respectively
        """
    # Read the csv files into dataframes
    data_list_fixed = pd.read_csv(fixed_csv, dtype=object,sep=';').values
    data_list_moving = pd.read_csv(moving_csv, dtype=object,sep=';').values

    data_fixed=[]
    data_moving=[]

    # Extract the paths from the dataframes and store them into separate lists
    for i in range(len(data_list_fixed)):
        data_fixed.append(data_list_fixed[i][0])
        data_moving.append(data_list_moving[i][0])

    return data_fixed, data_moving