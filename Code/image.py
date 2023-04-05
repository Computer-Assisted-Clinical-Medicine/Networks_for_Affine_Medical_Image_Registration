import numpy as np
import SimpleITK as sitk
import config as cfg

def get_data_info(sitk_image):
    """
    Extracts information about a given SimpleITK image and stores it in a dictionary.

    Args:
        sitk_image (SimpleITK.Image): SimpleITK image to extract information from.

    Returns:
        data_info: dictionary containing information about the image.

    """
    data_info = {}
    data_info['orig_spacing'] = sitk_image.GetSpacing()
    data_info['orig_origin'] = sitk_image.GetOrigin()
    data_info['orig_direction'] = sitk_image.GetDirection()
    data_info['orig_size'] = sitk_image.GetSize()
    data_info['orig_type'] = sitk_image.GetPixelID()
    return data_info

def np_array_to_itk_image(array, data_info, do_resample, out_type=sitk.sitkUInt8, background_value=0,
                          interpolator=sitk.sitkNearestNeighbor):
    """
    Converts a NumPy array to a SimpleITK image and sets its metadata.

    Args:
    array (numpy.ndarray): NumPy array to convert to SimpleITK image.
    data_info (dict): Dictionary containing metadata information for the image.
    do_resample (bool): Whether to resample the image or not.
    out_type (SimpleITK.PixelIDValueEnum): Output pixel type of the image (default: sitk.sitkUInt8).
    background_value (float): Value to use for background pixels (default: 0).
    interpolator (SimpleITK.InterpolatorEnum): Interpolation method to use during resampling (default: sitk.sitkNearestNeighbor).

    Returns:
    SimpleITK.Image: SimpleITK image with the specified metadata.

    """
    image = sitk.GetImageFromArray(array)
    image = sitk.Cast(image, out_type)

    if do_resample:
        image.SetSpacing(data_info['res_spacing'])
        image.SetOrigin(data_info['orig_origin'])
        image.SetDirection(data_info['res_direction'])
        image = sitk.Resample(image, data_info['orig_size'], sitk.Transform(3, sitk.sitkIdentity),
                              interpolator, data_info['orig_origin'], data_info['orig_spacing'],
                              data_info['orig_direction'], background_value, out_type)
    else:
        image.SetSpacing(data_info['orig_spacing'])
        image.SetOrigin(data_info['orig_origin'])
        image.SetDirection(data_info['orig_direction'])

    return image

def resample_sitk_image(data, target_info, data_background_value=-1000, do_adapt_resolution=True, label=None, label_background_value=0, do_augment=False, max_rotation_augment=0.07, max_resultion_augment=0.02):
    '''!
    resamples <tt>data</tt> and <tt>label</tt> image using simple Simple ITK

    @param data <em>ITK image,  </em> input data
    @param data_info, current meta information of data
    @param do_adapt_resolution <em>bool,  </em> only if True resolution is changed
    @param label <em>ITK image,  </em> same size as <tt>data</tt>
    @param do_augment <em>bool,  </em> enables data augmentation for training

    All initial image information is taken from the <tt>data</tt> image.
    The target spacing and target direction are set in config.py.
    If <tt>do_augment</tt> is true, a random rotation around the craniocaudal axis is added.
    The extend of the rotation is restricted by cgf.max_rotation_augment.

    @return resampled <tt>data</tt> and <tt>label</tt> as ITK images
    '''

    data_info = get_data_info(data)

    if do_augment and do_adapt_resolution:
        aug_target_spacing = [target_info['target_spacing'][0] + np.random.uniform(
            target_info['target_spacing'][0] * -max_resultion_augment,
            target_info['target_spacing'][0] * max_resultion_augment),
                              target_info['target_spacing'][1] + np.random.uniform(
                                  target_info['target_spacing'][1] * -max_resultion_augment,
                                  target_info['target_spacing'][1] * max_resultion_augment),
                              target_info['target_spacing'][2]]
        if cfg.print_details:
            print('           Augment Spacing:', data_info['orig_spacing'], target_info['target_spacing'], ':', aug_target_spacing)
        target_info['target_spacing'] = aug_target_spacing


    # how much world space is covered by the image
    x_extend = data_info['orig_spacing'][0] * data_info['orig_size'][0]
    y_extend = data_info['orig_spacing'][1] * data_info['orig_size'][1]
    z_extend = data_info['orig_spacing'][2] * data_info['orig_size'][2]
    #print('  In Extend: ', x_extend, y_extend, z_extend)

    if do_adapt_resolution:
        # size of the output image, so input space is covered with new resolution
        out_size = (target_info['target_size'][0], target_info['target_size'][1],target_info['target_size'][2])

        # When resampling, the origin has to be changed,
        # otherwise the patient will not be in the image center afterwards
        out_x_extend = target_info['target_spacing'][0] * out_size[0]
        out_y_extend = target_info['target_spacing'][1] * out_size[1]
        out_z_extend = target_info['target_spacing'][2] * out_size[2]

        # shift by half the difference in extend
        x_diff = (x_extend - out_x_extend) / 2
        y_diff = (y_extend - out_y_extend) / 2
        z_diff = (z_extend - out_z_extend) / 2

        out_spacing = target_info['target_spacing']

    else:
        out_size = data_info['orig_size']
        out_spacing = data_info['orig_spacing']

    # fix the direction
    if target_info['target_direction'] != data_info['orig_direction']:
        if target_info['target_direction'][0] > data_info['orig_direction'][0]:
            data_origin_x = data_info['orig_origin'][0] - x_extend
        else:
            data_origin_x = data_info['orig_origin'][0]
        if target_info['target_direction'][4] > data_info['orig_direction'][4]:
            data_origin_y = data_info['orig_origin'][1] - y_extend
        else:
            data_origin_y = data_info['orig_origin'][1]

        target_origin = (data_origin_x, data_origin_y, data_info['orig_origin'][2])
    else:
        target_origin = data_info['orig_origin']

    if do_adapt_resolution:
        # Multiply with the direction to shift according to the system axes.
        out_origin = (target_origin[0] + (x_diff * target_info['target_direction'][0]), target_origin[1] + (y_diff * target_info['target_direction'][4]), target_origin[2] + (z_diff * target_info['target_direction'][8]))
    else:
        out_origin = target_origin

    # if augmentation is on, do random translation and rotation
    if do_augment:
        transform = sitk.Euler3DTransform()
        # rotation center is center of the image center in world coordinates
        rotation_center = (data_info['orig_origin'][0] + (data_info['orig_direction'][0] * x_extend / 2),
                           data_info['orig_origin'][1] + (data_info['orig_direction'][4] * y_extend / 2),
                           data_info['orig_origin'][2] + (data_info['orig_direction'][8] * z_extend / 2))
        transform.SetCenter(rotation_center)
        # apply a random rotation around the z-axis
        rotation = np.random.uniform(np.pi * -max_rotation_augment, np.pi * max_rotation_augment)
        if cfg.print_details:
            print('           Augment Rotation:', rotation)
        transform.SetRotation(0, 0, rotation)
        transform.SetTranslation((0, 0, 0))
    else:
        transform = sitk.Transform(3, sitk.sitkIdentity)
    # data: linear resampling, fill outside with air

    new_data = sitk.Resample(data, out_size, transform, sitk.sitkLinear, out_origin, out_spacing,
                             target_info['target_direction'], data_background_value, target_info['target_type_image'])

    if label is not None:
        # label: nearest neighbor resampling, fill with background
        new_label = sitk.Resample(label, out_size, transform, sitk.sitkNearestNeighbor, out_origin, out_spacing,
                                  target_info['target_direction'], label_background_value, target_info['target_type_label'])
        return new_data, new_label
    else:
        return new_data



