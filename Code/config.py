"""!
@file config.py
Sets the parameters for configuration
"""
import SimpleITK as sitk
import os
from enum import Enum

#Insert the paths to your folders here
path_fixed = ""
path_moving = ""
path_seg_fixed = ""
path_seg_moving = ""
path_result = ""
path_pretrained = ""

experiment_name = "T_BenchmarkCNN/"

logs_path = os.path.join(path_result, experiment_name)

#Insert the paths to your csv files here
fixed_csv=path_fixed+"../../csv/CT.csv"
train_fixed_csv = logs_path + 'train_fixed_img.csv'
vald_fixed_csv = logs_path + 'vald_fixed_img.csv'
test_fixed_csv = logs_path + 'test_fixed_img.csv'

moving_csv=path_fixed+"../../csv/MR.csv"
train_moving_csv = logs_path + 'train_moving_img.csv'
vald_moving_csv = logs_path + 'vald_moving_img.csv'
test_moving_csv = logs_path+ 'test_moving_img.csv'

fixed_seg_csv=path_fixed+"../../csv/CT_seg.csv"
train_fixed_seg_csv = logs_path + 'train_fixed_seg.csv'
vald_fixed_seg_csv = logs_path + 'vald_fixed_seg.csv'
test_fixed_seg_csv = logs_path + 'test_fixed_seg.csv'

moving_seg_csv=path_fixed+"../../csv/MR_seg.csv"
train_moving_seg_csv = logs_path + 'train_moving_seg.csv'
vald_moving_seg_csv = logs_path + 'vald_moving_seg.csv'
test_moving_seg_csv = logs_path+ 'test_moving_seg.csv'

class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1

normalizing_method = NORMALIZING.WINDOW

number_of_vald = 5
num_train_files =-1

width=256
height=256
numb_slices=64

training=True

# Resampling
adapt_resolution = True
target_spacing = [2.0, 2.0, 4.0]
target_size = [256,256, 64]
target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # make sure all images are oriented equally
target_type_image = sitk.sitkFloat32
target_type_label = sitk.sitkUInt8
data_background_value = -1000
label_background_value = 0
max_rotation = 0

# Preprocessing
norm_min_v = -1024
norm_max_v = 1024
norm_eps = 1e-5

norm_min_v_fixed = -1024
norm_max_v_fixed = 1024
norm_min_v_moving = 0
norm_max_v_moving = 255
norm_min_v_label = 0
norm_max_v_label = 1

intervall_max=1

print_details=False
