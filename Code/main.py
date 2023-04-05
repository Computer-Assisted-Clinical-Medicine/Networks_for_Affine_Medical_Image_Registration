import os
os.environ['PATH'] += ';C:\Program Files\cudnn\cudnn-11.2\cuda/bin/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import datetime
import voxelmorph as vxm
import distutils.dir_util
import pandas as pd
from tensorflow.keras.models import save_model

import config as cfg
import util
import evaluation
import loadsave
from buildmodel import buildmodel
from datagenerator import DataGenerator, get_test_images
import processing

pretrained_model_name = 'weights.h5'

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

distutils.dir_util.mkpath(cfg.logs_path)

def training(f, architecture, losses, loss_weights, learning_rate, nb_epochs, batch_size, seed=42):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training = True

    traindata_files_fixed = pd.read_csv(cfg.train_fixed_csv, dtype=object).values
    traindata_files_moving = pd.read_csv(cfg.train_moving_csv, dtype=object).values

    valdata_files_fixed = pd.read_csv(cfg.vald_fixed_csv, dtype=object).values
    valdata_files_moving = pd.read_csv(cfg.vald_moving_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)

    print('inshape:', inshape)
    vxm_model=buildmodel(architecture, inshape)

    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    vxm_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)

    traindata_files_seg_fixed = pd.read_csv(cfg.train_fixed_seg_csv, dtype=object).values
    traindata_files_seg_moving = pd.read_csv(cfg.train_moving_seg_csv, dtype=object).values

    valdata_files_seg_fixed = pd.read_csv(cfg.vald_fixed_seg_csv, dtype=object).values
    valdata_files_seg_moving = pd.read_csv(cfg.vald_moving_seg_csv, dtype=object).values

    training_batch_generator = DataGenerator(traindata_files_fixed, cfg.path_fixed, traindata_files_moving, cfg.path_moving,
                                             batch_size, inshape, traindata_files_seg_fixed, traindata_files_seg_moving)
    validation_batch_generator = DataGenerator(valdata_files_fixed, cfg.path_fixed, valdata_files_moving, cfg.path_moving,
                                               batch_size, inshape, valdata_files_seg_fixed, valdata_files_seg_moving)


    log_dir = cfg.logs_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #Different callbacks
    callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                              restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.logs_path + 'checkpoint',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss', mode='min',
                                                                   save_best_only=True)

    callback = [callback_earlystopping, model_checkpoint_callback, callback_tensorboard]

    validation_steps = cfg.number_of_vald // batch_size
    vxm_model.fit(training_batch_generator, validation_data=validation_batch_generator,
                         validation_steps=validation_steps,
                         epochs=nb_epochs, batch_size=batch_size,
                         callbacks=[callback], verbose=2);

    vxm_model.save_weights(cfg.logs_path + str(f) + '/weights.h5')
    try:
        save_model(vxm_model, cfg.logs_path + str(f) + "/" )
    except:
        print("model save failed")


def pretrained(f, architecture, losses, loss_weights, learning_rate, nb_epochs, batch_size, seed=42):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training = True

    traindata_files_fixed = pd.read_csv(cfg.train_fixed_csv, dtype=object).values
    traindata_files_moving = pd.read_csv(cfg.train_moving_csv, dtype=object).values

    valdata_files_fixed = pd.read_csv(cfg.vald_fixed_csv, dtype=object).values
    valdata_files_moving = pd.read_csv(cfg.vald_moving_csv, dtype=object).values

    inshape = (cfg.height, cfg.width, cfg.numb_slices)

    print('inshape:', inshape)

    # vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model = buildmodel(architecture, inshape)
    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    vxm_model.load_weights(cfg.path_pretrained + str(f) + "/" + pretrained_model_name)
    vxm_model.trainable = True

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
    #vxm_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)

    traindata_files_seg_fixed = pd.read_csv(cfg.train_fixed_seg_csv, dtype=object).values
    traindata_files_seg_moving = pd.read_csv(cfg.train_moving_seg_csv, dtype=object).values

    valdata_files_seg_fixed = pd.read_csv(cfg.vald_fixed_seg_csv, dtype=object).values
    valdata_files_seg_moving = pd.read_csv(cfg.vald_moving_seg_csv, dtype=object).values

    training_batch_generator = DataGenerator(traindata_files_fixed, cfg.path_fixed, traindata_files_moving, cfg.path_moving,
                                                 batch_size, inshape, traindata_files_seg_fixed, traindata_files_seg_moving)
    validation_batch_generator = DataGenerator(valdata_files_fixed, cfg.path_fixed, valdata_files_moving, cfg.path_moving,
                                                   batch_size, inshape, valdata_files_seg_fixed, valdata_files_seg_moving)

    log_dir = cfg.logs_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Different callbacks
    callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                              restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.logs_path + 'checkpoint',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss', mode='min',
                                                                   save_best_only=True)

    callback = [callback_earlystopping, model_checkpoint_callback, callback_tensorboard]

    validation_steps = cfg.number_of_vald // batch_size

    vxm_model.fit(training_batch_generator, validation_data=validation_batch_generator,
                  validation_steps=validation_steps,
                  epochs=nb_epochs, batch_size=batch_size,
                  callbacks=[callback], verbose=2);

    vxm_model.save_weights(cfg.logs_path + str(f) + '/weights.h5')
    try:
        save_model(vxm_model, cfg.logs_path + str(f) + "/")
    except:
        print("model save failed")

def apply(f, architecture, batch_size,seed=42):
    '''!
    predict images, (segmentations, ) displacementfields for test files
    use shape fixed image
    '''
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training=False

    test_data_fixed = pd.read_csv(cfg.test_fixed_csv, dtype=object).values
    test_data_moving = pd.read_csv(cfg.test_moving_csv, dtype=object).values

    test_data_seg_moving = pd.read_csv(cfg.test_moving_seg_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)
    vxm_model = buildmodel(architecture, inshape)

    print("Test file size: ", len(test_data_fixed))
    vxm_model.load_weights(cfg.logs_path + str(f) + '/weights.h5')

    predict_path = cfg.logs_path+'predict/'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    for i in range(int(len(test_data_fixed) / batch_size)):
        test_images = get_test_images(test_data_fixed[i][0], test_data_moving[i][0])
        predictions = vxm_model.predict(test_images, steps=1,
                                        batch_size=batch_size, verbose=2)

        filename_fixed = test_data_fixed[i][0][2:-1]
        filename_moving = test_data_moving[i][0][2:-1]

        cfg.orig_filepath=cfg.path_fixed+filename_fixed

        filename_seg_moving = test_data_seg_moving[i][0][2:-1]

        processing.warp_img_fixed_size(predictions[-1], predict_path, filename_moving)
        processing.warp_seg_fixed_size(predictions[-1], predict_path + "/seg/", filename_seg_moving)

def evaluate(f):
    '''!
    evaluate predicted images with metrics
    used shape fixed image
    '''

    np.random.seed(42)

    cfg.training = False

    test_data_fixed = pd.read_csv(cfg.test_fixed_csv, dtype=object).values
    test_data_moving = pd.read_csv(cfg.test_moving_csv, dtype=object).values

    test_data_seg_fixed = pd.read_csv(cfg.test_fixed_seg_csv, dtype=object).values
    test_data_seg_moving = pd.read_csv(cfg.test_moving_seg_csv, dtype=object).values

    distutils.dir_util.mkpath(cfg.logs_path + 'eval/')
    eval_file_path = cfg.logs_path+'eval/' + 'eval-'+str(f)+'.csv'

    header_row = evaluation.make_csv_header()
    util.make_csv_file(eval_file_path, header_row)

    predict_path = cfg.logs_path + 'predict/'

    for i in range(len(test_data_fixed)):
        filename_fixed = test_data_fixed[i][0][2:-1]
        filename_moving = test_data_moving[i][0][2:-1]

        filename_seg_fixed = test_data_seg_fixed[i][0][2:-1]
        filename_seg_moving = test_data_seg_moving[i][0][2:-1]
        try:
            result_metrics = {}
            result_metrics['FILENAME_FIXED'] = filename_fixed
            result_metrics['FILENAME_MOVING'] = filename_moving

            result_metrics = evaluation.evaluate_prediction(result_metrics, predict_path,
                                                            ('moved' + '_' + filename_moving),
                                                            cfg.path_fixed_resized, filename_fixed,
                                                            filename_seg_fixed, filename_seg_moving)

            util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)
            print('Finished Evaluation for ' , filename_fixed , 'and' , filename_moving)
        except RuntimeError as err:
            print("    !!! Evaluation of " , filename_fixed , 'and' , filename_moving , ' failed',err)


    #read csv
    header=pd.read_csv(eval_file_path, dtype=object,sep=';')
    header = header.columns.values
    values = pd.read_csv(eval_file_path, dtype=object,sep=';').values
    np_values = np.empty(values.shape)

    result_metrics['FILENAME_FIXED'] = 'min'
    result_metrics['FILENAME_MOVING'] = ' '

    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.min(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'mean'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.average(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'max'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.max(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

def experiment(data_fixed, data_moving, architecture, losses, loss_weights, learning_rate, nb_epochs,
                 batch_size, is_training, is_pretrained, is_apply, is_evaluate):
    k_fold=5
    np.random.seed(42)
    all_indices = np.random.permutation(range(0, len(data_fixed)))
    test_folds = np.array_split(all_indices, k_fold)

    data_seg_fixed, data_seg_moving = loadsave.getdatalist_from_csv(cfg.fixed_seg_csv, cfg.moving_seg_csv)

    for f in range(k_fold):
        test_indices = test_folds[f]
        remaining_indices = np.random.permutation(np.setdiff1d(all_indices, test_folds[f]))
        vald_indices = remaining_indices[:cfg.number_of_vald]
        train_indices = remaining_indices[cfg.number_of_vald:]

        train_files_fixed=np.empty(len(train_indices), dtype = "S70")
        train_files_moving=np.empty(len(train_indices), dtype = "S70")
        vald_files_fixed=np.empty(len(vald_indices), dtype = "S70")
        vald_files_moving=np.empty(len(vald_indices), dtype = "S70")
        test_files_fixed=np.empty(len(test_indices), dtype = "S70")
        test_files_moving=np.empty(len(test_indices), dtype = "S70")

        train_files_seg_fixed = np.empty(len(train_indices), dtype="S70")
        train_files_seg_moving = np.empty(len(train_indices), dtype="S70")
        vald_files_seg_fixed = np.empty(len(vald_indices), dtype="S70")
        vald_files_seg_moving = np.empty(len(vald_indices), dtype="S70")
        test_files_seg_fixed = np.empty(len(test_indices), dtype="S70")
        test_files_seg_moving = np.empty(len(test_indices), dtype="S70")

        for i in range(len(train_indices)):
            train_files_fixed[i] = data_fixed[train_indices[i]]
            train_files_moving[i] = data_moving[train_indices[i]]
            train_files_seg_fixed[i] = data_seg_fixed[train_indices[i]]
            train_files_seg_moving[i] = data_seg_moving[train_indices[i]]

        for i in range(len(vald_indices)):
            vald_files_fixed[i] = data_fixed[vald_indices[i]]
            vald_files_moving[i] = data_moving[vald_indices[i]]
            vald_files_seg_fixed[i] = data_seg_fixed[vald_indices[i]]
            vald_files_seg_moving[i] = data_seg_moving[vald_indices[i]]

        for i in range(len(test_indices)):
            test_files_fixed[i] = data_fixed[test_indices[i]]
            test_files_moving[i] = data_moving[test_indices[i]]
            test_files_seg_fixed[i] = data_seg_fixed[test_indices[i]]
            test_files_seg_moving[i] = data_seg_moving[test_indices[i]]


        np.savetxt(cfg.train_fixed_csv, train_files_fixed, fmt='%s', header='path')
        np.savetxt(cfg.vald_fixed_csv, vald_files_fixed, fmt='%s', header='path')
        np.savetxt(cfg.test_fixed_csv, test_files_fixed, fmt='%s', header='path')

        np.savetxt(cfg.train_moving_csv, train_files_moving, fmt='%s', header='path')
        np.savetxt(cfg.vald_moving_csv, vald_files_moving, fmt='%s', header='path')
        np.savetxt(cfg.test_moving_csv, test_files_moving, fmt='%s', header='path')

        np.savetxt(cfg.train_fixed_seg_csv, train_files_seg_fixed, fmt='%s', header='path')
        np.savetxt(cfg.vald_fixed_seg_csv, vald_files_seg_fixed, fmt='%s', header='path')
        np.savetxt(cfg.test_fixed_seg_csv, test_files_seg_fixed, fmt='%s', header='path')

        np.savetxt(cfg.train_moving_seg_csv, train_files_seg_moving, fmt='%s', header='path')
        np.savetxt(cfg.vald_moving_seg_csv, vald_files_seg_moving, fmt='%s', header='path')
        np.savetxt(cfg.test_moving_seg_csv, test_files_seg_moving, fmt='%s', header='path')

        cfg.num_train_files = train_indices.size

        print(str(train_indices.size) + ' train cases, ' + str(test_indices.size) + ' test cases, '
              + str(vald_indices.size) + ' vald cases')

        distutils.dir_util.mkpath(cfg.logs_path+'/'+ str(f))

        if is_training:
                training(f, architecture, losses, loss_weights, learning_rate, nb_epochs, batch_size,
                         seed=f)

        if is_pretrained:
                pretrained(f, architecture, losses, loss_weights, learning_rate, nb_epochs, batch_size,
                          seed=f)
        if is_apply:
                apply(f, architecture, batch_size, seed=f)

        if is_evaluate:
                evaluate(f)

    if is_evaluate:
        evaluation.combine_evaluation_results_from_folds(cfg.logs_path+'eval/')
        evaluation.combine_evaluation_results_in_file(cfg.logs_path+'eval/')
        evaluation.make_boxplot_graphic(cfg.logs_path+'eval/')

#main

is_training = True
is_pretrained = False
is_apply = True
is_evaluate= True

learning_rate=1e-3

data_fixed, data_moving= loadsave.getdatalist_from_csv(cfg.fixed_csv, cfg.moving_csv)

cfg.height=256
cfg.width=256
cfg.numb_slices=64

batch_size=1

nb_epochs = 100

architecture='BenchmarkCNN'

losses=[vxm.losses.MSE().loss,vxm.losses.Grad('l2').loss, vxm.losses.Dice().loss]
loss_weights = [0, 0, 1] #Train only with Dice loss

distutils.dir_util.mkpath(cfg.logs_path)

experiment(data_fixed, data_moving, architecture, losses, loss_weights, learning_rate, nb_epochs, batch_size,
           is_training=is_training, is_pretrained=is_pretrained, is_apply=is_apply, is_evaluate=is_evaluate)


