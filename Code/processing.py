import tensorflow as tf
import voxelmorph as vxm
import numpy as np
import distutils.dir_util
from scipy.ndimage import zoom
import SimpleITK as sitk
import config as cfg
import loadsave
import image

def img_to_nparray(img):
    """
        Converts a SimpleITK image object to a numpy array.

        Args:
            img (SimpleITK image object): The image.

        Returns:
            image data (numpy array).
        """
    data = sitk.GetArrayFromImage(img)
    data = np.moveaxis(data, 0, -1)
    return data

def nparray_to_img(img):
    """
        Converts a numpy array to a SimpleITK image object.

        Args:
            img (numpy array).

        Returns:
            image data (SimpleITK image object).
        """

    img = np.moveaxis(img, -1, 0)
    img = sitk.GetImageFromArray(img)
    return img

def warp_img_fixed_size(displacementfield, path_result, filename_moving):
    """
        Warps a moving image using a displacement field. Moved image is then saved.

        Args:
            displacementfield: A displacement field numpy array.
            path_result: A string representing the path to save the warped image.
            filename_moving: A string representing the filename of the moving image.

        Returns:
            None.
        """

    # Load the moving image and displacementfield.
    moving_np=loadsave.load_image_data_info_fixed_image(cfg.path_moving,filename_moving)
    displacementfield= resize_dispfield(displacementfield)

    if cfg.print_details:
        print(" warp_img_fixed_size",moving_np.shape)
        print(" warp_img_fixed_size displacementfield", displacementfield.shape)

    #pad moving image
    moving_shape_orig=moving_np.shape
    pad_y=displacementfield.shape[0]-moving_np.shape[0]
    pad_x=displacementfield.shape[1]-moving_np.shape[1]
    pad_z=displacementfield.shape[2]-moving_np.shape[2]

    if pad_x<0:
        pad_x = 0
    if pad_y < 0:
        pad_y = 0
    if pad_z < 0:
        pad_z=0

    pad_amount = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (pad_z // 2, pad_z - pad_z // 2))
    moving_np = np.pad(moving_np, pad_amount, 'constant')

    displacementfield = np.expand_dims(displacementfield, axis=0)

    moving_np = np.expand_dims(moving_np, axis=0)
    moving_np = np.expand_dims(moving_np, axis=4)

    # Build the transformer layer.
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # Warp the moving image using the displacement field.
    moved_np = spatial_transformer(
        [tf.convert_to_tensor(moving_np), tf.convert_to_tensor(displacementfield)])

    if cfg.print_details:
        print("image_warped shape", moved_np.shape)

    # Save the warped image (moved image)
    moved_np = moved_np[0, :, :, :, 0]
    moved_np = moved_np[pad_y // 2:pad_y // 2 + moving_shape_orig[0], pad_x // 2:pad_x // 2 + moving_shape_orig[1],
        pad_z // 2:pad_z // 2 + moving_shape_orig[2]]
    loadsave.save_image(moved_np, path_result, "moved_"+filename_moving, use_orig_file_info=True, do_resample=False)

def resize_dispfield(displacementfield):
    """
    Resizes a given displacement field to match the target spacing specified in the configuration file.

    Args:
        displacementfield: numpy array representing the displacement field to be resized.

    Returns:
         displacementfield_resized: numpy array representing the resized displacement field.

    """
    orig_img = sitk.ReadImage(cfg.orig_filepath)
    data_info = image.get_data_info(orig_img)

    # Set remaining data info fields
    data_info['res_spacing'] = cfg.target_spacing
    data_info['res_origin'] = [0.0, 0.0, 0.0]
    data_info['res_direction'] = cfg.target_direction

    displacementfield_resized=[]

    for i in range(len(data_info['res_spacing'])):
        zoom_factor = (data_info['res_spacing'][0]/data_info['orig_spacing'][0],data_info['res_spacing'][1]/data_info['orig_spacing'][1],data_info['res_spacing'][2]/data_info['orig_spacing'][2])
        displacementfield_resized_1 = zoom(displacementfield[0,:,:,:,i], zoom_factor)

        # also change magnitude of displacementfield according to the zoom factor for resizing
        displacementfield_resized_1 = displacementfield_resized_1 * zoom_factor[i]
        displacementfield_resized_1 = np.expand_dims(displacementfield_resized_1, axis=3)
        if i==0:
            displacementfield_resized = displacementfield_resized_1
        else:
            displacementfield_resized = np.append(displacementfield_resized, displacementfield_resized_1, axis=3)

    return displacementfield_resized


def seg_binary(img_np,threshold=0.5):
    """
        Converts an input image array to binary, with a given threshold.

        Args:
        - img_np (numpy array): input image array, with shape (height, width, channels)
        - threshold (float): threshold value for segmentation (default is 0.5)

        Returns:
        - img_bin (numpy array): binary image array, with shape (height, width, channels)
        """
    img_bin = np.where(img_np < threshold, 0, 1)
    return img_bin

def warp_seg_fixed_size(displacementfield, predict_path, filename_seg_moving):
    """
        Applies a warp to a segmentation image using a displacement field. Moved segmentation is then saved.

        Args:
        - displacementfield (numpy array): the displacement field for the warp
        - predict_path (string): path to save the warped segmentation
        - filename_seg_moving (string): the filename of the moving segmentation

        Returns:
        - None
        """
    distutils.dir_util.mkpath(predict_path)

    # Load the moving segmentation image
    seg_np = loadsave.load_image_data_info_fixed_image(cfg.path_seg_moving, filename_seg_moving)
    seg_np = seg_binary(seg_np)

    displacementfield= resize_dispfield(displacementfield)

    # pad seg
    seg_shape_orig = seg_np.shape
    pad_y = displacementfield.shape[0] - seg_np.shape[0]
    pad_x = displacementfield.shape[1] - seg_np.shape[1]
    pad_z = displacementfield.shape[2] - seg_np.shape[2]

    if pad_x<0:
        pad_x = 0
    if pad_y < 0:
        pad_y = 0
    if pad_z < 0:
        pad_z=0

    pad_amount = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (pad_z // 2, pad_z - pad_z // 2))

    seg_np = np.pad(seg_np, pad_amount, 'constant')

    displacementfield = np.expand_dims(displacementfield, axis=0)

    seg_np = np.expand_dims(seg_np, axis=0)
    seg_np = np.expand_dims(seg_np, axis=4)

    # Build the transformer layer
    spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

    # Warp the moving segmentation using the displacement field
    seg_warped = spatial_transformer([tf.convert_to_tensor(seg_np, dtype=tf.float32),
                                      tf.convert_to_tensor(displacementfield, dtype=tf.float32)])

    seg_warped = seg_warped[0, :, :, :, 0]

    # crop image
    seg_warped = seg_warped[pad_y // 2:pad_y // 2 + seg_shape_orig[0], pad_x // 2:pad_x // 2 + seg_shape_orig[1],
            pad_z // 2:pad_z // 2 + seg_shape_orig[2]]
    seg_warped = seg_binary(seg_warped)
    loadsave.save_image(seg_warped, predict_path, "seg_"+filename_seg_moving, use_orig_file_info=True, do_resample=False)
