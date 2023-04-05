import numpy as np
import tensorflow
import config as cfg
import loadsave

class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, fixed_filenames, path_fixed, moving_filenames, path_moving, batch_size,shape, fixed_seg_filenames, moving_seg_filenames):
        self.fixed_filenames = fixed_filenames
        self.moving_filenames= moving_filenames
        self.path_fixed = path_fixed
        self.path_moving = path_moving
        self.batch_size = batch_size
        self.shape=shape
        self.fixed_seg_filenames = fixed_seg_filenames
        self.moving_seg_filenames = moving_seg_filenames

    def __len__(self):
        return (np.ceil(len(self.fixed_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_fixed = self.fixed_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_moving = self.moving_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        vol_shape = self.shape
        ndims = len(self.shape)
        zero_phi = np.zeros([self.batch_size, *vol_shape, ndims])

        fixed_images = np.array([loadsave.load_image(self.path_fixed, file_name[0][2:-1]) for file_name in batch_fixed])
        if cfg.print_details:
            print(batch_fixed[0][0][2:-1], batch_moving[0][0][2:-1])
        moving_images = np.array([loadsave.load_image(self.path_moving, file_name[0][2:-1]) for file_name in batch_moving])

        fixed_images = fixed_images[0, :, :, :]
        moving_images = moving_images[0, :, :, :]

        batch_seg_fixed = self.fixed_seg_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_seg_moving = self.moving_seg_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        fixed_seg = np.array([loadsave.load_image(cfg.path_seg_fixed, file_name[0][2:-1]) for file_name in batch_seg_fixed])
        moving_seg = np.array([loadsave.load_image(cfg.path_seg_moving, file_name[0][2:-1]) for file_name in batch_seg_moving])

        fixed_seg = fixed_seg[0, :, :, :]
        moving_seg = moving_seg[0, :, :, :]

        fixed_images = np.expand_dims(fixed_images, axis=0)
        moving_images = np.expand_dims(moving_images, axis=0)
        fixed_seg = np.expand_dims(fixed_seg, axis=0)
        moving_seg = np.expand_dims(moving_seg, axis=0)

        moving_images = np.expand_dims(moving_images, axis=4)
        fixed_images = np.expand_dims(fixed_images, axis=4)
        moving_seg = np.expand_dims(moving_seg, axis=4)
        fixed_seg = np.expand_dims(fixed_seg, axis=4)

        inputs = [moving_images, fixed_images, moving_seg]
        outputs = [fixed_images, zero_phi, fixed_seg]
        return (inputs, outputs)


def get_test_images(fixed_filename,moving_filename):

    fixed_image = loadsave.load_image(cfg.path_fixed, fixed_filename[2:-1])
    moving_image = loadsave.load_image(cfg.path_moving, moving_filename[2:-1])

    fixed_image = np.expand_dims(fixed_image, axis=0)
    moving_image = np.expand_dims(moving_image, axis=0)
    moving_image = np.expand_dims(moving_image, axis=4)
    fixed_image = np.expand_dims(fixed_image, axis=4)
    inputs = [moving_image, fixed_image]

    return (inputs)
