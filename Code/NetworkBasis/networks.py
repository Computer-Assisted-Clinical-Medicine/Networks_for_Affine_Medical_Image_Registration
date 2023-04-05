import voxelmorph as vxm
from classification_models_3D.keras import Classifiers
from voxelmorph import layers as vxm_layers
from voxelmorph.tf.modelio import LoadableModel, store_config_args

from NetworkBasis.densenet import DenseBlocks
from NetworkBasis.layers import *
from NetworkBasis.swin import *
from NetworkBasis.C2FVIT import *


# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class Model_BenchmarkCNN(LoadableModel):

    @store_config_args
    def __init__(self,inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        ndim = 3
        unet_input_features = 2
        src_feats = 1
        trg_feats = 1

        nb_features = [512, 256, 128, 64, 32, 12]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        UpSampling = getattr(KL, 'UpSampling%dD' % ndim)
        Net, preprocess_input = Classifiers.get('resnet18')
        net_input = KL.concatenate(input_model.outputs, name='input_concat')
        base_model = Net(input_shape=(*inshape, unet_input_features), input_tensor=net_input, weights=None, include_top=False)
        x=base_model.output

        for i in range(len(nb_features)-1):
            x = _conv_block(x, nb_features[i])
            x = UpSampling(size=(2,) * ndim)(x)
        x = _conv_block(x, nb_features[-1])
        output_avg_3d = tf.keras.layers.GlobalAveragePooling3D(data_format='channels_last')(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, output_avg_3d])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='BenchmarkCNN', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff=output_avg_3d

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Huetal(LoadableModel):
    """
    Hu, Y., Modat, M., Gibson, E., Li, W., Ghavami, N., Bonmati, E., Wang, E., Bandula, S., Moore, C.M., Emberton,
    M., Ourselin, S., Noble, J.A., Barratt, D.C., Vercauteren, T., 2018. Weakly-supervised convolutional neural networks
    for multimodal image registration. Medical Image Analysis 49, 1–13. doi:10.1016/j.media.2018.07.002.
    https://github.com/NifTK/NiftyNet
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        ndim = 3
        src_feats = 1
        trg_feats = 1

        nb_features = [4, 8, 16, 32, 64]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        input = KL.concatenate(input_model.outputs, name='input_concat')

        res_1 = DownBlock(inshape=input.shape, n_output_chns=nb_features[0], kernel_size=7, i="1")(input)
        res_2 = DownBlock(inshape=res_1[0].shape, n_output_chns=nb_features[1], i="2")(res_1[0])
        res_3 = DownBlock(inshape=res_2[0].shape, n_output_chns=nb_features[2], i="3")(res_2[0])
        res_4 = DownBlock(inshape=res_3[0].shape, n_output_chns=nb_features[3], i="4")(res_3[0])
        Conv = getattr(KL, 'Conv%dD' % ndim)

        conv_5 = Conv(nb_features[4], kernel_size=3, padding='same', kernel_initializer='he_normal', strides=1,
                      use_bias=False, name='Conv5')(res_4[0])

        convolved = tf.keras.layers.BatchNormalization(axis=-1)(conv_5)
        last = KL.ReLU(0.2, name='Conv5_activation')(convolved)

        x = KL.Flatten()(last)
        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer_aff')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Huetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

class Model_Guetal(LoadableModel):
    """
    Gu, D., Liu, G., Tian, J., Zhan, Q., 2019. Two-Stage Unsupervised Learning Method for Affine and Deformable Medical
    Image Registration, in: 2019 IEEE International Conference on Image Processing (ICIP), pp. 1332–1336.
    doi:10.1109/ICIP.2019.8803794
    """

    @store_config_args
    def __init__(self,inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features=[4, 8, 16, 32, 64]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        x=net_input
        for i in range(len(nb_features)):
            x= KL.Conv3D(nb_features[i],kernel_size=3, padding="SAME", name="conv"+str(i))(x)
            x= KL.MaxPooling3D(2, name="conv"+str(i)+"_pooling")(x)
            x = KL.LeakyReLU(0.2, name="conv" + str(i) + "_activation")(x)
        x = KL.Flatten()(x)

        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Guetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Shenetal(LoadableModel):
    """
    Shen, Z., Han, X., Xu, Z., Niethammer, M., 2019. Networks for Joint Affine and Non-Parametric Image Registration,
    in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE Computer Society, Los Alamitos,
    CA, USA. pp. 4219–4228. doi:10.1109/CVPR.2019. 00435.
    https://github.com/uncbiag/easyreg/tree/master/easyreg
    """

    @store_config_args
    def __init__(self, inshape, apply=False):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1
        nb_features=[16, 16, 4]

        if apply:
            nb_steps=7
        else:
            nb_steps=5

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        Model=Model_multistep_unet(inshape, nb_features)
        spatial_transformer=SpatialTransformer_with_disp(name='transformer')
        moving_image=source
        for i in range(nb_steps):
            _, affine_param = Model([moving_image, target])
            if i>0:
                affine_param=self.update_affine_param(affine_param, affine_param_last)
            affine_param_last = affine_param
            moving_image, disp_tensor = spatial_transformer(
                [source, affine_param])

        outputs = [moving_image, disp_tensor]

        super().__init__(name='Shenetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moving_image
        self.references.pos_flow = disp_tensor

    def update_affine_param(self,cur_af, last_af):
        """
            https://github.com/uncbiag/easyreg/blob/5e93d76813580ab936ed4c682b1b2711cf1bac39/easyreg/affine_net.py#L138
            update the current affine parameter A2 based on last affine parameter A1
             A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2, results in the composed affine parameter A3=(A2A1, A2*b1+b2)
            :param cur_af: current affine parameter
            :param last_af: last affine parameter
            :return: composed affine parameter A3
        """
        cur_af = KL.Reshape((4, 3))(cur_af)
        last_af = KL.Reshape((4, 3))(last_af)
        updated_af1=tf.linalg.matmul(cur_af[:, :3, :], last_af[:, :3, :])
        updated_af2=tf.expand_dims(cur_af[:, 3, :] + tf.squeeze(tf.matmul(cur_af[:, :3, :], tf.transpose(last_af[:, 3:, :], (0, 2, 1))), 2), axis=1)
        updated_af= tf.concat([updated_af1, updated_af2], 1)
        updated_af = KL.Flatten()(updated_af)
        return updated_af

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_multistep_unet(LoadableModel):
    """
    Shen, Z., Han, X., Xu, Z., Niethammer, M., 2019. Networks for Joint Affine and Non-Parametric Image Registration,
    in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE Computer Society, Los Alamitos,
    CA, USA. pp. 4219–4228. doi:10.1109/CVPR.2019. 00435.
    https://github.com/uncbiag/easyreg/tree/master/easyreg
    """

    @store_config_args
    def __init__(self,
                 inshape, nb_features):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
            nb_features: encoder.
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        d1_m = _conv_block_custom(source, nb_features[0], strides=1, name="conv1_m", batch_norm=False, activation="relu")
        d1_t = _conv_block_custom(target, nb_features[0], strides=1, name="conv1_t", batch_norm=False, activation="relu")
        d1 = KL.concatenate([d1_m,d1_t])
        d2 = KL.MaxPooling3D(2)(d1)
        d4 = _conv_block_custom(d2, nb_features[1], strides=2, name="conv2", batch_norm=False, activation="relu")
        d8 = KL.MaxPooling3D(2)(d4)
        d16 = _conv_block_custom(d8, nb_features[2], strides=2, name="conv3", batch_norm=False, activation="relu")
        d32 = KL.MaxPooling3D(2)(d16)
        d32 = KL.Flatten()(d32)
        fc1 = KL.Dense(32, name="fc_1", activation=KL.ReLU())(d32)
        parameters_aff = KL.Dense(12, name="fc_2")(fc1)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, parameters_aff]

        super().__init__(name='multistep_unet', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Zhaoetal(LoadableModel):
    """
    Zhao, S., Lau, T., Luo, J., Chang, E.I.C., Xu, Y., 2020. Unsupervised 3D End-to-End Medical Image Registration With
    Volume Tweening Network. IEEE Journal of Biomedical and Health Informatics 24, 1394–1404.
    doi:10.1109/jbhi.2019.2951024.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [16, 32, 64, 128,128,256,256,512,512]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        conv1 = _conv_block_custom(net_input, nb_features[0], strides=2, name="conv1")
        conv2 = _conv_block_custom(conv1, nb_features[1], strides=2, name="conv2")
        conv3 = _conv_block_custom(conv2, nb_features[2], strides=2, name="conv3")
        conv4_1 = _conv_block_custom(conv3, nb_features[3], strides=2, name="conv4_1")
        conv4_2 = _conv_block_custom(conv4_1, nb_features[4], strides=1, name="conv4_2")
        conv5_1 = _conv_block_custom(conv4_2, nb_features[5], strides=2, name="conv5_1")
        conv5_2 = _conv_block_custom(conv5_1, nb_features[6], strides=1, name="conv5_2")
        conv6_1 = _conv_block_custom(conv5_2, nb_features[7], strides=2, name="conv6_1")
        conv6_2 = _conv_block_custom(conv6_1, nb_features[8], strides=1, name="conv6_2")
        x = KL.Flatten()(conv6_2)

        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Zhaoetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Luoetal(LoadableModel):
    """
    Luo, G., Chen, X., Shi, F., Peng, Y., Xiang, D., Chen, Q., Xu, X., Zhu, W., Fan, Y., 2020. Multimodal affine
    registration for ICGA and MCSL fundus images of high myopia. Biomedical Optics Express 11. doi:10.1364/BOE. 393178.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        unet_input_features = 2
        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')


        inshape_1 = (*inshape, unet_input_features)
        Net, preprocess_input = Classifiers.get("resnet18")
        base_model = Net(input_shape=inshape_1, input_tensor=net_input, weights=None, classes=512, include_top=False)
        fc_1 = KL.Flatten()(base_model.output)
        fc_1 = KL.Dense(512, name="fc_1")(fc_1)
        fc_1 = KL.LeakyReLU(0.2, name="fc_1_activation")(fc_1)
        fc_2 = KL.Dense(256, name="fc_2")(fc_1)
        fc_2 = KL.LeakyReLU(0.2, name="fc_2_activation")(fc_2)
        fc_3 = KL.Dense(64, name="fc_3")(fc_2)
        x = KL.LeakyReLU(0.2, name="fc_3_activation")(fc_3)
        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Luoetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Tangetal(LoadableModel):
    """
    Tang, K., Li, Z., Tian, L., Wang, L., Zhu, Y., 2020. ADMIR–Affine and Deformable Medical Image Registration for
    Drug-Addicted Brain Images. IEEE Access 8, 70960–70968. doi:10.1109/ACCESS.2020.2986829.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [8, 16, 32, 64, 128, 256]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        conv1 = _conv_block_custom(net_input, nb_features[0], strides=2, name="conv1", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv2 = _conv_block_custom(conv1, nb_features[1], strides=2, name="conv2", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv3 = _conv_block_custom(conv2, nb_features[2], strides=2, name="conv3", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv4 = _conv_block_custom(conv3, nb_features[3], strides=2, name="conv4", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv5 = _conv_block_custom(conv4, nb_features[4], strides=2, name="conv5", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv6 = _conv_block_custom(conv5, nb_features[5], strides=2, name="conv6", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        x = KL.Flatten()(conv6)
        parameters_aff_1 = KL.Dense(9)(x)
        parameters_aff_translation = KL.Dense(3)(x)
        parameters_aff_1 = KL.Dropout(0.3)(parameters_aff_1)
        parameters_aff_1 = tf.reshape(parameters_aff_1, shape=(1,3,3))
        parameters_aff_translation = KL.Dropout(0.3)(parameters_aff_translation)
        parameters_aff=KL.concatenate([parameters_aff_1,tf.expand_dims(parameters_aff_translation,axis=-1)],-1)
        parameters_aff = KL.Flatten()(parameters_aff)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Tangetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Zengetal(LoadableModel):
    """
    Zeng, Q., Fu, Y., Tian, Z., Lei, Y., Zhang, Y., Wang, T., Mao, H., Liu, T., Curran, W., Jani, A., Patel, P.,
    Yang, X., 2020. Label-driven MRI-US registration using weakly-supervised learning for MRI-guided prostate
    radiotherapy. Physics in Medicine & Biology 65. doi:10.1088/1361-6560/ab8cd6.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        ndim = 3 # 2 or 3 for 2D or 3D version
        src_feats = 1
        trg_feats = 1

        nb_features = [128, 256, 512, 1024, 12]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        conv1 = _conv_block_custom(net_input, nb_features[0], strides=2, name="conv1", batch_norm=True, batch_norm_mode="before_act", activation="relu", ndims=ndim)
        conv2 = _conv_block_custom(conv1, nb_features[1], strides=2, name="conv2", batch_norm=True, batch_norm_mode="before_act", activation="relu", ndims=ndim)
        conv3 = _conv_block_custom(conv2, nb_features[2], strides=2, name="conv3", batch_norm=True, batch_norm_mode="before_act", activation="relu", ndims=ndim)
        conv4 = _conv_block_custom(conv3, nb_features[3], strides=2, name="conv4", batch_norm=True, batch_norm_mode="before_act", activation="relu", ndims=ndim)
        conv5 = _conv_block_custom(conv4, nb_features[4], strides=2, name="conv5", batch_norm=False, activation="none", ndims=ndim)
        x = KL.Flatten()(conv5)
        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Zengetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_XChenetal(LoadableModel):
    """
    Chen, X., Meng, Y., Zhao, Y., Williams, R., Vallabhaneni, S.R., Zheng, Y., 2021b. Learning Unsupervised
    Parameter-Specific Affine Transformation for Medical Images Registration. pp. 24–34.
    doi:10.1007/978-3-030-87202-1\_3.
    https://github.com/xuuuuuuchen/PASTA/blob/master/models.py
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: encoder and decoder.
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [64, 32, 32, 32, 16, 16]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        source_r = KL.Reshape(inshape)(source)
        target_r = KL.Reshape(inshape)(target)

        source_r = tf.expand_dims(source_r, axis=1)
        target_r = tf.expand_dims(target_r, axis=1)

        unit_num=2
        if unit_num == 1:
            cs_level = [5]
        elif unit_num == 2:
            cs_level = [4, 5]
        elif unit_num == 3:
            cs_level = [3, 4, 5]
        elif unit_num == 4:
            cs_level = [2, 3, 4, 5]
        elif unit_num == 5:
            cs_level = [1, 2, 3, 4, 5]
        elif unit_num == 6:
            cs_level = [0, 1, 2, 3, 4, 5]

        def Conv3dBlock(filters, x):

            dpr = 0.2
            x = KL.Conv3D(
                filters, kernel_size=3,
                padding='same',
                data_format='channels_first',
                use_bias=False)(x)

            x = KL.BatchNormalization()(x)
            x = KL.ReLU(negative_slope=0.01)(x)
            x = KL.Dropout(dpr, seed=1)(x)

            x = KL.Conv3D(
                filters, kernel_size=3,
                padding='same',
                data_format='channels_first',
                use_bias=False)(x)

            x = KL.BatchNormalization()(x)
            x = KL.ReLU(negative_slope=0.01)(x)
            x = KL.Dropout(dpr, seed=2)(x)

            x = KL.AveragePooling3D(pool_size=2, data_format='channels_first')(x)

            return x

        class CrossStitch(Layer):
            # basic parameter setting
            def __init__(self, input_shape):
                super(CrossStitch, self).__init__()
                self.shape = np.prod(input_shape[1:])
                self.input_shape_1 = self.shape
                self.input_shape_2 = self.shape
                # self.output_shape = [input_shape[1],input_shape[2],input_shape[3]]

            # in cross-stitch network: [xa,xb]*[papameter]=[xa',xb'], the detail refer to the paper
            def build(self, input_shape):
                shape = self.input_shape_1 + self.input_shape_2
                self.cross_stitch = self.add_weight(
                    shape=(shape, shape),
                    initializer=tf.initializers.identity(),
                    name='CrossStitch')
                self.built = True

            # conduct implement of the detailed algorithm calculation
            # inputs represent the output of upper layer, such as x=Dense(parameter)(inputs)
            def call(self, inputs):
                x1 = KL.Reshape((self.shape,))(inputs[0])
                x2 = KL.Reshape((self.shape,))(inputs[1])

                inputss = tf.concat((x1, x2), axis=1)
                output = tf.matmul(inputss, self.cross_stitch)
                output1 = output[:, :self.input_shape_1]
                output2 = output[:, self.input_shape_2:]
                # print("output1.shape",output1.shape)
                # print("inputs[0].shape",inputs[0].shape)

                s1 = inputs[0].shape[1]
                s2 = inputs[0].shape[2]
                s3 = inputs[0].shape[3]
                s4 = inputs[0].shape[4]

                output1 = tf.reshape(
                    output1,
                    shape=[tf.shape(inputs[0])[0], s1, s2, s3, s4])

                output2 = tf.reshape(
                    output2,
                    shape=[tf.shape(inputs[0])[0], s1, s2, s3, s4])
                # print("output1.shape",output1.shape)
                return [output1, output2]

        x1 = source_r
        x2 = target_r

        for i in range(len(nb_features)):
            # print("cs")
            # print(num_channels[i])
            # print(x1.shape)
            # print(x2.shape)
            x1 = Conv3dBlock(nb_features[i], x1)
            x2 = Conv3dBlock(nb_features[i], x2)
            # print(x1.shape)
            # print(x2.shape)
            if i in cs_level:
                [x1, x2] = CrossStitch(x1.shape)([x1, x2])

        x = KL.concatenate([x1, x2],axis=1)
        x = KL.Flatten()(x)

        PASTA = True
        if PASTA == True:

            # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
            # Rotation_2D_Range = np.arange(-30,30.01,10)
            # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
            # Scale_2D_Range = np.arange(0.90,1.101,0.05)

            initial_transformation_params = [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0]

            initial_transformation_params = tf.constant_initializer(value=initial_transformation_params)

            transformation_params = KL.Dense(15, bias_initializer=initial_transformation_params,
                                          name='transformation_params')(x)

            max_value = 1.0
            Threshold = 0.0

            transformation_params = KL.ReLU(negative_slope=0.01, max_value=max_value, threshold=Threshold)(transformation_params)

            branch_outputs = []

            for i in range(15):
                out = KL.Lambda(lambda x: x[:, i], name="Splitting_" + str(i))(transformation_params)

                if i == 0 or i == 1 or i == 2:
                    y_max, y_min = 0.201, -0.201
                    Pasta_Para = KL.Lambda(lambda x: tf.expand_dims((y_max - y_min) * x + y_min, 1),
                                        name='Mapping_' + str(i))(out)

                    branch_outputs.append(Pasta_Para)
                elif i == 3 or i == 4 or i == 5:
                    y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                    # y_max, y_min = 0.34900, -0.34900
                    # y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                    # y_max, y_min = 30.01, -30.01
                    Pasta_Para = KL.Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min), 1),
                                        name='Mapping_' + str(i))(out)

                    branch_outputs.append(Pasta_Para)

                elif i > 5 and i < 12:
                    y_max, y_min = 0.051, -0.051
                    Pasta_Para = KL.Lambda(lambda x: tf.expand_dims((y_max - y_min) * x + y_min, 1),
                                        name='Mapping_' + str(i))(out)

                    branch_outputs.append(Pasta_Para)

                elif i > 5:
                    y_max, y_min = 1.051, 0.951
                    Pasta_Para = KL.Lambda(lambda x: tf.expand_dims((y_max - y_min) * x + y_min, 1),
                                        name='Mapping_' + str(i))(out)

                    branch_outputs.append(Pasta_Para)

            PASTA_15 = KL.Concatenate(axis=1, name='PASTA_15')(branch_outputs)

            parameters_aff = KL.Lambda(lambda x: Combining_Affine_Para3D(x), name='AFF_12')([source, PASTA_15])

        else:

            initial_affine_matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]

            initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

            parameters_aff = KL.Dense(12,bias_initializer=initial_affine_matrix)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='XChenetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Gaoetal(LoadableModel):
    """
    Gao, X., Van Houtte, J., Chen, Z., Zheng, G., 2021. DeepASDM: a Deep Learning Framework for Affine and Deformable
    Image Registration Incorporating a Statistical Deformation Model, in: 2021 IEEE EMBS International Conference on
    Biomedical and Health Informatics (BHI), pp. 1–4. doi:10.1109/BHI50953.2021.9508553.
    """

    @store_config_args
    def __init__(self,
                 inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')


        conv1_1 = _conv_block_custom(net_input, nb_features[0], strides=2, name="conv1_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv1_2 = _conv_block_custom(conv1_1, nb_features[1], strides=1, name="conv1_2", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv2_1 = _conv_block_custom(conv1_2, nb_features[2], strides=2, name="conv2_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv2_2 = _conv_block_custom(conv2_1, nb_features[3], strides=1, name="conv2_2", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv3_1 = _conv_block_custom(conv2_2, nb_features[4], strides=2, name="conv3_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv3_2 = _conv_block_custom(conv3_1, nb_features[5], strides=1, name="conv3_2", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv4_1 = _conv_block_custom(conv3_2, nb_features[6], strides=2, name="conv4_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv4_2 = _conv_block_custom(conv4_1, nb_features[7], strides=1, name="conv4_2", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv5_1 = _conv_block_custom(conv4_2, nb_features[8], strides=2, name="conv5_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv5_2 = _conv_block_custom(conv5_1, nb_features[9], strides=1, name="conv5_2", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv6_1 = _conv_block_custom(conv5_2, nb_features[10], strides=2, name="conv6_1", batch_norm=True, batch_norm_mode="before_act", activation="leakyrelu")
        conv6_2 = _conv_block_custom(conv6_1, nb_features[11], strides=1, name="conv6_2", batch_norm=True,batch_norm_mode="before_act", activation="leakyrelu")
        x = KL.Flatten()(conv6_2)

        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Gaoetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        #self.references.net_model = net
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Roelofs(LoadableModel):
    """
    Roelofs, T.J.T., 2021. Deep Learning-Based Affine and Deformable 3D Medical Image Registration. Master’s thesis.
    Aalto University. Espoo, Finnland.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        unet_input_features = 2
        src_feats = 1
        trg_feats = 1
        inshape_1 = (*inshape, unet_input_features)

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        Net, preprocess_input = Classifiers.get("resnet18")

        base_model = Net(input_shape=inshape_1,input_tensor=net_input, weights=None, classes=12)
        parameters_aff = base_model.output

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Roelofs', inputs=base_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.base_model = base_model
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Shaoetal(LoadableModel):
    """
    Shao, W., Banh, L., Kunder, C.A., Fan, R.E., Soerensen, S.J., Wang, J.B., Teslovich, N.C., Madhuripan, N.,
    Jawahar, A., Ghanouni, P., Brooks, J.D., Sonn, G.A., Rusu, M., 2021. ProsRegNet: A deep learning framework for
    registration of MRI and histopathology images of the prostate. Medical Image Analysis 68, 101919.
    doi:https://doi.org/10.1016/j.media.2020. 101919.
    Website https://github.com/pimed/ProsRegNet
    """

    @store_config_args
    def __init__(self,
                 inshape, normalize_features=True, normalize_matches=True):
        """
        Parameters:
            inshape: Input shape. e.g. (256, 256, 64)
            normalize_features: A boolean indicating whether to normalize the image features (default is True).
            normalize_matches: A boolean indicating whether to normalize the feature matches (default is True).
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        Net, preprocess_input = Classifiers.get('resnet101')
        base_model = Net(input_shape=(*inshape, 1), weights=None,
                         include_top=False)
        x = base_model.layers[-37].output #after layer 3 (add_29 (Add) of stage 3)
        FeatureExtraction = tf.keras.Model(inputs=base_model.input, outputs=x)

        feature_A=FeatureExtraction(source)
        feature_B=FeatureExtraction(target)

        if normalize_features:
            feature_A=FeatureL2NormLayer()(feature_A)
            feature_B=FeatureL2NormLayer()(feature_B)

        correlation = FeatureCorrelation()(feature_A, feature_B)
        if normalize_matches:
            correlation =FeatureL2NormLayer()(KL.ReLU()(correlation))

        # FeatureRegression
        # do regression to tnf parameters theta
        theta = FeatureRegression()(correlation)

        adjust = tf.constant([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0], dtype=tf.float32)
        adjust = tf.cast(adjust, dtype=tf.float32)
        theta = 0.1 * theta + adjust
        theta = tf.cast(theta, dtype=tf.float32)

        parameters_aff = KL.Flatten()(theta)

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = SpatialTransformer_with_disp(name='transformer_aff')([source, parameters_aff])

        # model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)
        outputs=[moved_image_tensor, disp_tensor]

        super().__init__(name='Shaoetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Zhuetal(LoadableModel):
    """
    Zhu, Z., Cao, Y., Chenchen, Q., Rao, Y., Di, L., Dou, Q., Ni, D., Wang, Y.,2021. Joint affine and deformable
    three-dimensional networks for brain MRI registration. Medical Physics 48. doi:10.1002/mp.14674.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [16, 16, 32, 32, 32, 32, 32]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        conv1 = _conv_block_custom(net_input, nb_features[0], strides=1, name="conv1", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv2 = _conv_block_custom(conv1, nb_features[1], strides=2, name="conv2", batch_norm=False, activation="None")
        conv3 = _conv_block_custom(conv2, nb_features[2], strides=1, name="conv3", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv4 = _conv_block_custom(conv3, nb_features[3], strides=2, name="conv4", batch_norm=False, activation="None")
        conv5 = _conv_block_custom(conv4, nb_features[4], strides=1, name="conv5", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")
        conv6 = _conv_block_custom(conv5, nb_features[5], strides=2, name="conv6", batch_norm=False, activation="None")
        conv7 = _conv_block_custom(conv6, nb_features[6], strides=1, name="conv7", batch_norm=True, batch_norm_mode="after_act", activation="leakyrelu")

        x = tf.keras.layers.GlobalAveragePooling3D(data_format='channels_last')(conv7)
        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Zhuetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Cheeetal_Venkataetal(LoadableModel):
    """
    Chee, E., Wu, Z., 2018. AIRNet: Self-Supervised Affine Registration for 3D Medical Images using Neural Networks.
    CoRR abs/1810.02583. URL: http://arxiv.org/abs/1810.02583, arXiv:1810.02583.

    Venkata, S.P., Duffy, B.A., Datta, K., 2022. An unsupervised deep learning method for affine registration of
    multi-contrast brain MR images. ISMRM 2022.
    """

    @store_config_args
    def __init__(self,
                 inshape, name):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
            name: Architecture: Cheeetal or Venkataetal
        """

        src_feats = 1
        trg_feats = 1

        if name == "Cheeetal":
            use_2d_filters = True
        elif name == "Venkataetal":
            use_2d_filters = False

        nb_features = [16, 8, 8, 8, 8, 8, 8]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        net=Enc_part_weight_sharing(input_model=input_model, nb_features=nb_features, use_2d_filters=use_2d_filters)
        x = KL.Flatten()(net.output)

        fc_1 = KL.Dense(1024, name="fc_1")(x)
        fc_1 = tf.keras.layers.BatchNormalization(axis=-1, name="fc_1_bn")(fc_1)
        fc_1 = KL.ReLU(name="fc_1_activation")(fc_1)
        fc_2 = KL.Dense(512, name="fc_2")(fc_1)
        fc_2 = tf.keras.layers.BatchNormalization(axis=-1, name="fc_2_bn")(fc_2)
        fc_2 = KL.ReLU(name="fc_2_activation")(fc_2)
        fc_3 = KL.Dense(128, name="fc_3")(fc_2)
        fc_3 = tf.keras.layers.BatchNormalization(axis=-1, name="fc_3_bn")(fc_3)
        fc_3 = KL.ReLU(name="fc_3_activation")(fc_3)
        fc_4 = KL.Dense(64, name="fc_4")(fc_3)
        fc_4 = tf.keras.layers.BatchNormalization(axis=-1, name="fc_4_bn")(fc_4)
        fc_4 = KL.ReLU(name="fc_4_activation")(fc_4)
        fc_5 = KL.Dense(12, name="fc_5")(fc_4)
        fc_5 = tf.keras.layers.BatchNormalization(axis=-1, name="fc_5_bn")(fc_5)
        parameters_aff = KL.ReLU(name="fc_5_activation")(fc_5)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Cheeetal_Venkataetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

class Model_deVosetal(LoadableModel):
    """
    de Vos, B.D., Berendsen, F.F., Viergever, M.A., Sokooti, H., Staring, M., Isgum, I., 2019. A deep learning framework
    for unsupervised affine and deformable image registration. Medical Image Analysis 52, 128–143.
    doi:10.1016/j.media.2018.11.010.
    """

    @store_config_args
    def __init__(self,
                 inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256, 256, 64)
        """

        src_feats = 1
        trg_feats = 1

        nb_features = [16, 32, 64, 128, 256]

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        model = tf.keras.Sequential()

        for i in range(len(nb_features)):
            model.add(KL.Conv3D(nb_features[i], kernel_size=3, padding='same', strides=1, name="conv_"+str(i)))
            model.add(KL.ReLU(name="activation_"+str(i)))
            if i < (len(nb_features)-1):
                model.add(KL.MaxPooling3D(2, name="pooling_"+ str(i)))

        model.add(KL.GlobalAveragePooling3D(data_format='channels_last'))

        f=model(target)
        m=model(source)

        x = KL.concatenate([f, m], name='Concat')

        x = KL.Flatten()(x)
        x = KL.Dense(32, activation=KL.LeakyReLU(0.2))(x)

        parameters_aff = KL.Dense(12)(x)

        parameters_aff = AffineTransformationsToMatrix(ndims=3, use_constraint=True, name='affineparameterstomatrix')(parameters_aff)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  #layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='deVosetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff = parameters_aff

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_deSilvaetal(LoadableModel):
    """
    de Silva, T., Chew, E.Y., Hotaling, N., Cukras, C.A., 2020. Deep-Learning based Multi-Modal Retinal Image
    Registration for Longitudinal Analysis of Patients with Age-related Macular Degeneration. Biomedical Optics
    Express 12. doi:10.1364/BOE.408573.
    """

    @store_config_args
    def __init__(self, inshape, normalize_features=True, normalize_matches=True):
        """
        Parameters:
            inshape: Input shape. e.g. (256, 256, 64)
            normalize_features: A boolean indicating whether to normalize the image features (default is True).
            normalize_matches: A boolean indicating whether to normalize the feature matches (default is True).
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        net_input = KL.concatenate(input_model.outputs, name='input_concat')

        Net, preprocess_input = Classifiers.get('vgg16')
        base_model = Net(input_shape=(*inshape, 1), weights=None,
                         include_top=False)
        x = base_model.layers[-5].output  # after layer 3 (add_29 (Add) of stage 3)
        FeatureExtraction = tf.keras.Model(inputs=base_model.input, outputs=x)

        f1 = FeatureExtraction(target)
        f2 = FeatureExtraction(source)

        if normalize_features:
            f1=FeatureL2NormLayer()(f1)
            f2=FeatureL2NormLayer()(f2)

        correlation = FeatureCorrelation()(f1, f2)
        if normalize_matches:
            correlation =FeatureL2NormLayer()(KL.ReLU()(correlation))

        #Regression
        conv1 = KL.Conv3D(128, kernel_size=3, padding="SAME", kernel_initializer='he_normal', strides=1,
                          name="conv1")(correlation)
        conv1 = tf.keras.layers.BatchNormalization(axis=-1, name="conv1_batch_norm")(conv1)
        conv1 = KL.Dropout(0.2)(conv1)
        conv1 = KL.ReLU(name="conv1_act")(conv1)
        conv2 = KL.Conv3D(64, kernel_size=3, padding="SAME", kernel_initializer='he_normal', strides=1,
                          name="conv2")(conv1)
        conv2 = tf.keras.layers.BatchNormalization(axis=-1, name="conv2_batch_norm")(conv2)
        conv2 = KL.Dropout(0.2)(conv2)
        conv2 = KL.ReLU(name="conv2_act")(conv2)
        conv2 = KL.Flatten()(conv2)

        parameters_aff = KL.Dense(12)(conv2)

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = SpatialTransformer_with_disp(name='transformer_aff')([source, parameters_aff])

        outputs=[moved_image_tensor, disp_tensor]

        super().__init__(name='deSilvaetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        # self.references.net_model = net
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_Waldkirch(LoadableModel):
    """
    Waldkirch, B.I., 2020. Methods for three-dimensional Registration of Multimodal Abdominal Image Data. Ph.D. thesis.
    Ruprecht Karl University of Heidelberg.
    """

    @store_config_args
    def __init__(self,
                 inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        nb_features = [[8, 8, 8, 8],[24, 24, 24, 12, 12, 12, 12]]

        # build core unet model and grab inputs
        unet= vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_features
        )

        output_avg_3d = tf.keras.layers.GlobalAveragePooling3D(data_format='channels_last')(unet.output)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, output_avg_3d])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Waldkirch', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor
        self.references.aff=output_avg_3d

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_JChenetal(LoadableModel):
    """
    Chen, J., Frey, E.C., He, Y., Segars, W.P., Li, Y., Du, Y., 2022. TransMorph: Transformer for unsupervised medical
    image registration. Medical Image Analysis 82, 102615. doi:https://doi.org/10.1016/j.media.2022.102615.
    https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        input = KL.concatenate(input_model.outputs, name='input_concat')

        out = SwinTransformerModel(
            model_name="swin_affine", include_top=False, num_classes=1000, img_size=inshape,
            window_size=16, embed_dim=12, depths=(1, 1, 2), num_heads=(1, 1, 2))(input)
        x5=KL.Flatten()(out)
        aff=KL.Dense(100)(x5)
        aff=KL.ReLU()(aff)
        aff = KL.Dense(3)(aff)
        scl = KL.Dense(100)(x5)
        scl = KL.ReLU()(scl)
        scl = KL.Dense(3)(scl)
        trans = KL.Dense(100)(x5)
        trans = KL.ReLU()(trans)
        trans = KL.Dense(3)(trans)
        shr = KL.Dense(100)(x5)
        shr = KL.ReLU()(shr)
        shr = KL.Dense(6)(shr)

        aff = clamp(aff, minn=-1.0, maxn=1.0)*math.pi
        scl = scl+1
        scl = clamp(scl, minn=0.0, maxn=5.0)
        shr = clamp(shr, minn=-1.0, maxn=1.0) * math.pi

        parameters_aff = AffineTransformationsToMatrix_swin()(source, aff, scl, trans, shr)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp_swin(name='transformer')

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='JChenetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_MokandChung(LoadableModel):
    """
    Mok, T.C.W., Chung, A.C.S., 2022. Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. 2022
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 20803–20812.
    https://github.com/cwmok/C2FViT
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        outputs=Model_C2FViT(img_size=inshape)(source,target)

        super().__init__(name='MokandChung', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = outputs[0]
        self.references.pos_flow = outputs[1]

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Model_C2FViT(LoadableModel):
    """
    Mok, T.C.W., Chung, A.C.S., 2022. Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. 2022
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 20803–20812.
    https://github.com/cwmok/C2FViT
    """

    #@store_config_args
    def __init__(self,
                 img_size, patch_size = [3, 7, 15], stride = [2, 4, 8], num_classes = 12, embed_dims = [256, 256, 256], num_heads = [2, 2, 2],
                 mlp_ratios = [2, 2, 2], qkv_bias = False, qk_scale = None, drop_rate = 0., attn_drop_rate = 0., norm_layer = tf.identity,
                 depths = [4, 4, 4], sr_ratios = [1, 1,1], num_stages = 3, linear = False):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
            nb_features: encoder and decoder.
        """
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        self.img_size = img_size
        self.stride = stride
        self.num_stages = num_stages

        self.patch_embed_xy_list = []
        self.stage_list = []
        self.head_list = []
        self.squeeze_list = []

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size,
                                            patch_size=patch_size[i],
                                            stride=stride[i],
                                            in_chans=2,
                                            embed_dim=embed_dims[i])
            self.patch_embed_xy_list.append(patch_embed)

            stage = tf.keras.Sequential(
                [Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
                       sr_ratio=sr_ratios[i], linear=linear) for _ in range(depths[i])]
            )
            self.stage_list.append(stage)

            head = tf.keras.Sequential(
                [
                    KL.Dense(embed_dims[i] // 2, use_bias=False),
                    KL.ReLU(),
                    KL.Dense(num_classes, use_bias=False),
                    KL.Activation("tanh")
                ]
            )
            self.head_list.append(head)

        for i in range(num_stages-1):
            squeeze = KL.Conv3D(embed_dims[i + 1], kernel_size=3, strides=1, padding="SAME", data_format='channels_first')
            self.squeeze_list.append(squeeze)

        self.avg_pool = KL.AveragePooling3D(2, 2,data_format="channels_first")
        self.affine_transform = AffineCOMTransform()
        self._init_weights()

    def __call__(self, source, target):
        source=KL.Reshape((1,*self.img_size))(source)
        target = KL.Reshape((1, *self.img_size))(target)

        x = source
        y = target
        B = x.shape[0]

        warpped_x_list = []
        affine_list = []

        x = self.image_pyramid(x, self.num_stages)
        y = self.image_pyramid(y, self.num_stages)

        for i in range(self.num_stages):
            if i == 0:
                xy = KL.Concatenate(axis=1)([x[i], y[i]])
            else:
                xy = KL.Concatenate(axis=1)([warpped_x_list[i - 1], y[i]])

            patch_embed_xy = self.patch_embed_xy_list[i]

            xy_patch_embed, H, W, D = patch_embed_xy(xy)
            H, W, D = xy.shape[2] // self.stride[i], xy.shape[3] // self.stride[i], xy.shape[4] // self.stride[i]

            if i > 0:
                xy_patch_embed = xy_patch_embed + xy_fea

            xy_fea = xy_patch_embed
            stage_block = self.stage_list[i]

            for blk in stage_block.layers:
                xy_fea = blk(xy_fea, H, W, D)

            head = self.head_list[i]
            # affine = head(xy_fea.mean(dim=1))
            affine = head(tf.math.reduce_mean(xy_fea, axis=1))
            affine_list.append(affine)

            # Used SpatialTransformer_with_disp (VoxelMorph) instead of F.grid_sample in AffineCOMTransform()
            spatial_transformer = SpatialTransformer_with_disp(name='transformer' + str(i))

            if i < self.num_stages - 1:
                affine_param = self.affine_transform(x[i + 1], affine)
                warpped_x, disp_tensor = spatial_transformer([KL.Permute((2, 3, 4, 1))(x[i + 1]), affine_param])
                moved_image_tensor = warpped_x
                warpped_x = KL.Permute((4, 1, 2, 3))(warpped_x)
                warpped_x_list.append(warpped_x)

                # xy_fea = KL.Reshape((1, Hp, Wp, Dp, -1))(xy_fea)
                xy_fea = KL.Reshape((H, W, D, -1))(xy_fea)
                xy_fea = KL.Permute((4, 1, 2, 3))(xy_fea)
                squeeze = self.squeeze_list[i]
                xy_fea = squeeze(xy_fea)
                xy_fea = KL.Reshape((-1, H * W * D))(xy_fea)
                xy_fea = KL.Permute((2, 1))(xy_fea)
            else:
                affine_param = self.affine_transform(x[i], affine)
                warpped_x, disp_tensor = spatial_transformer([KL.Permute((2, 3, 4, 1))(x[i]), affine_param])
                moved_image_tensor = warpped_x
                warpped_x = KL.Permute((4, 1, 2, 3))(warpped_x)
                warpped_x_list.append(warpped_x)

        return [moved_image_tensor, disp_tensor]

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, tf.keras.layers.Dense):
                tf.keras.initializers.TruncatedNormal(stddev=0.02)(m.weight)
                if isinstance(m, tf.keras.layers.Dense) and m.bias is not None:
                    tf.keras.initializers.Zeros()(m.bias)
            elif isinstance(m, tf.keras.layers.LayerNormalization):
                tf.keras.initializers.Zeros()(m.beta)
                tf.keras.initializers.Ones()(m.gamma)
            elif isinstance(m, tf.keras.layers.Conv3D):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.filters
                fan_out //= m.groups
                tf.keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(2.0 / fan_out))(m.weights)
                #if m.bias is not None: #bias_initializer='zeros' is default value, see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
                #    tf.keras.initializers.Zeros()(m.bias)

    def image_pyramid(self, x, level=3):
        out = [x]
        for i in range(level - 1):
            x = self.avg_pool(x)
            out.append(x)

        return out[::-1]

class Model_Hasenstabetal(LoadableModel):
    """
    Hasenstab, K.A., Cunha, G.M., Higaki, A., Ichikawa, S., Wang, K., Delgado, T., Brunsing, R.L., Schlein, A.,
    Bittencourt, L.K., Schwartzman, A., Fowler, K.J., Hsiao, A., Sirlin, C.B., 2019. Fully automated convolutional
    neural network-based affine algorithm improves liver registration and lesion co-localization on hepatobiliary phase
    T1-weighted MR images. Eur Radiol Exp. 3(1):43. doi:10.1186/s41747-019-0120-7.
    """

    @store_config_args
    def __init__(self, inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
        """

        src_feats = 1
        trg_feats = 1

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        input = KL.concatenate(input_model.outputs, name='input_concat')
        x = KL.Flatten()(input)
        parameters_aff = KL.Dense(12)(x)

        # build transformer layer
        spatial_transformer = SpatialTransformer_with_disp(name='transformer')  # layers_AS

        # warp the moving image with the transformer
        moved_image_tensor, disp_tensor = spatial_transformer([source, parameters_aff])

        outputs = [moved_image_tensor, disp_tensor]

        super().__init__(name='Hasenstabetal', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        # self.references.unet_model = unet
        self.references.y_source = moved_image_tensor
        self.references.pos_flow = disp_tensor

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class SemiSupervisedSegModel(LoadableModel):
    """
    Network for (semi-supervised) registration between two images.
    Adapted from: https://github.com/voxelmorph/voxelmorph
    """

    @store_config_args
    def __init__(self, inshape, architecture):
        """
        Parameters:
            inshape: Input shape. e.g. (256,256,64)
            architecture: architecture of the network
        """

        if architecture == "BenchmarkCNN":
            vxm_model = Model_BenchmarkCNN(inshape)
        elif architecture == "Huetal":
            vxm_model=Model_Huetal(inshape)
        elif architecture == "Guetal":
            vxm_model=Model_Guetal(inshape)
        elif architecture == "Shenetal":
            vxm_model = Model_Shenetal(inshape, apply=False)
        elif architecture == "Zhaoetal":
            vxm_model=Model_Zhaoetal(inshape)
        elif architecture == "Luoetal":
            vxm_model = Model_Luoetal(inshape)
        elif architecture == "Tangetal":
            vxm_model = Model_Tangetal(inshape)
        elif architecture == "Zengetal":
            vxm_model = Model_Zengetal(inshape)
        elif architecture == 'XChenetal':
            vxm_model = Model_XChenetal(inshape)
        elif architecture == 'Gaoetal':
            vxm_model = Model_Gaoetal(inshape)
        elif architecture == 'Roelofs':
            vxm_model = Model_Roelofs(inshape)
        elif architecture == "Shaoetal":
            vxm_model = Model_Shaoetal(inshape)
        elif architecture == "Zhuetal":
            vxm_model = Model_Zhuetal(inshape)
        elif architecture == "Cheeetal" or architecture == "Venkataetal":
            vxm_model = Model_Cheeetal_Venkataetal(inshape, architecture)
        elif architecture == "deVosetal":
            vxm_model = Model_deVosetal(inshape)
        elif architecture == "deSilvaetal":
            vxm_model = Model_deSilvaetal(inshape)
        elif architecture == "Waldkirch":
            vxm_model = Model_Waldkirch(inshape)
        elif architecture =="JChenetal":
            vxm_model = Model_JChenetal(inshape)
        elif architecture == "MokandChung":
            vxm_model = Model_MokandChung(inshape)
        elif architecture == 'Hasenstabetal':
            vxm_model = Model_Hasenstabetal(inshape)

        seg_src = tf.keras.Input(shape=(*inshape, 1))
        y_seg = vxm_layers.SpatialTransformer(interp_method='linear', indexing='ij', name='seg_transformer')(
            [seg_src, vxm_model.references.pos_flow])

        inputs = vxm_model.inputs + [seg_src]
        outputs = vxm_model.outputs + [y_seg]
        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.pos_flow = vxm_model.references.pos_flow

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm_layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class Enc_part_weight_sharing(tf.keras.Model):
    """
    Chee, E., Wu, Z., 2018. AIRNet: Self-Supervised Affine Registration for 3D Medical Images using Neural Networks.
    CoRR abs/1810.02583. URL: http://arxiv.org/abs/1810.02583, arXiv:1810.02583.

    Venkata, S.P., Duffy, B.A., Datta, K., 2022. An unsupervised deep learning method for affine registration of
    multi-contrast brain MR images. ISMRM 2022.
    """
    def __init__(self,
                 input_model=None,
                 nb_features=None,
                 use_2d_filters=False):
        """
        Parameters:
            input_model: Input model
            nb_features: Convolutional features
            use_2d_filters: In the initial part of the encoder, 2D filters are used in the convolutional and pooling layers.
        """

        if use_2d_filters:
            Conv1 = KL.Conv2D(nb_features[0], kernel_size=3, padding='same', kernel_initializer='he_normal', strides=1, name="conv_1")
            Conv2 = KL.Conv2D(nb_features[1], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_2")
            Conv3 = KL.Conv2D(nb_features[2], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_3")
            Conv4 = KL.Conv2D(nb_features[3], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_4")
            dim = 2
            pooling_size = (2,2,1)
            strides = (2,2,1)
        else:
            Conv1 = KL.Conv3D(nb_features[0], kernel_size=3, padding='same', kernel_initializer='he_normal',strides=1, name="conv_1")
            Conv2 = KL.Conv3D(nb_features[1], kernel_size=1, padding='same', kernel_initializer='he_normal',strides=1, name="conv_2")
            Conv3 = KL.Conv3D(nb_features[2], kernel_size=1, padding='same', kernel_initializer='he_normal',strides=1, name="conv_3")
            Conv4 = KL.Conv3D(nb_features[3], kernel_size=1, padding='same', kernel_initializer='he_normal',strides=1, name="conv_4")
            dim = 3
            pooling_size = 2
            strides = 2
        Conv5 = KL.Conv3D(nb_features[4], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_5")
        Conv6 = KL.Conv3D(nb_features[5], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_6")
        Conv7 = KL.Conv3D(nb_features[6], kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1, name="conv_7")


        out_1_1=Conv1(input_model.output[0])
        out_1_1 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_1)
        out_1_1 = KL.ReLU(name="conv_1_1_activation")(out_1_1)

        out_1_1 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_1')(out_1_1)
        Dense1 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=1, shape=out_1_1.shape[1:], name='Dense1',dim=dim)
        out_1_1 = Dense1(out_1_1)

        out_2_1 = Conv1(input_model.output[1])
        out_2_1 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_1)
        out_2_1 = KL.ReLU(name="conv_2_1_activation")(out_2_1)

        out_2_1 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_1')(out_2_1)
        out_2_1 = Dense1(out_2_1)

        out_1_2 = Conv2(out_1_1)
        out_1_2 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_2)
        out_1_2 = KL.ReLU(name="conv_1_2_activation")(out_1_2)

        out_1_2 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_2')(out_1_2)
        Dense2 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=2, shape=out_1_2.shape[1:], name='Dense2',dim=dim)
        out_1_2  = Dense2(out_1_2)

        out_2_2 = Conv2(out_2_1)
        out_2_2 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_2)
        out_2_2 = KL.ReLU(name="conv_2_2_activation")(out_2_2)

        out_2_2 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_2')(out_2_2)
        out_2_2 = Dense2(out_2_2)

        out_1_3 = Conv3(out_1_2)
        out_1_3 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_3)
        out_1_3 = KL.ReLU(name="out_1_3_activation")(out_1_3)

        out_1_3 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_3')(out_1_3)
        Dense3 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=4, shape=out_1_3.shape[1:],name='Dense3',dim=dim)
        out_1_3 = Dense3(out_1_3)

        dim = 3

        out_2_3 = Conv3(out_2_2)
        out_2_3 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_3)
        out_2_3 = KL.ReLU(name="conv_2_3_activation")(out_2_3)

        out_2_3 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_3')(out_2_3)
        out_2_3 = Dense3(out_2_3)

        out_1_4 = Conv4(out_1_3)
        out_1_4 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_4)
        out_1_4 = KL.ReLU(name="conv_1_4_activation")(out_1_4)

        out_1_4 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_4')(out_1_4)
        Dense4 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=8, shape=out_1_4.shape[1:], name='Dense4',dim=dim)
        out_1_4 = Dense4(out_1_4)

        out_2_4 = Conv4(out_2_3)
        out_2_4 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_4)
        out_2_4 = KL.ReLU(name="conv_2_4_activation")(out_2_4)

        out_2_4 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_4')(out_2_4)
        out_2_4 = Dense4(out_2_4)

        pooling_size = 2
        strides = 2

        out_1_5 = Conv5(out_1_4)
        out_1_5 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_5)
        out_1_5 = KL.ReLU(name="out_1_5_activation")(out_1_5)

        out_1_5 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_5')(out_1_5)
        Dense5 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=16, shape=out_1_5.shape[1:], name='Dense5',dim=dim)
        out_1_5 = Dense5(out_1_5)

        out_2_5 = Conv5(out_2_4)
        out_2_5 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_5)
        out_2_5 = KL.ReLU(name="conv_2_5_activation")(out_2_5)

        out_2_5 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_5')(out_2_5)
        out_2_5 = Dense5(out_2_5)

        out_1_6 = Conv6(out_1_5)
        out_1_6 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_6)
        out_1_6 = KL.ReLU(name="conv_1_6_activation")(out_1_6)

        out_1_6 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_6')(out_1_6)
        Dense6 = DenseBlocks(growth_rate=8, nb_filter=8, nb_layers=32, shape=out_1_6.shape[1:], name='Dense6',dim=dim)
        out_1_6 = Dense6(out_1_6)

        out_2_6 = Conv6(out_2_5)
        out_2_6 = tf.keras.layers.BatchNormalization(axis=-1)(out_2_6)
        out_2_6 = KL.ReLU(name="conv_2_6_activation")(out_2_6)

        out_2_6 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_6')(out_2_6)
        out_2_6 = Dense6(out_2_6)

        out_1_7 = Conv7(out_1_6)
        out_1_7 = tf.keras.layers.BatchNormalization(axis=-1)(out_1_7)
        out_1_7 = KL.ReLU(name="conv_1_7_activation")(out_1_7)

        out_1_7 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_1_7')(out_1_7)

        out_2_7 = Conv7(out_2_6)
        out_2_7= tf.keras.layers.BatchNormalization(axis=-1)(out_2_7)
        out_2_7 = KL.ReLU(0.2, name="conv_2_7_activation")(out_2_7)

        out_2_7 = KL.MaxPooling3D(pooling_size, strides=strides, name='pooling_2_7')(out_2_7)

        x1 = KL.Flatten()(out_1_7)
        x2 = KL.Flatten()(out_2_7)
        last = KL.concatenate([x1, x2], name='concat')

        super().__init__(inputs=input_model.output, outputs=last)

class DownBlock(tf.keras.Model):
    """
    adapted to TF 2 from https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/downsample_res_block.py
    Block with ResUnit

        Consists of::
            (inputs)--conv_0-o-conv_1--conv_2-+-(conv_res)--down_sample--
                             |                |
                             o----------------o
        conv_0, conv_res is also returned for feature forwarding purpose

    """
    def __init__(self, input_model=None, inshape=None,
             n_output_chns=4,
             kernel_size=3,
             maxpool=2,
             type_string='bn_acti_conv',
             i="1"):

        ndims=3
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            input = KL.Input(shape=inshape[1:], name='net_input')
            model_inputs = [input]
        else:
            input = KL.concatenate(input_model.outputs, name='net_input_concat')
            model_inputs = input_model.inputs

        Conv = getattr(KL, 'Conv%dD' % ndims)
        conv_0 = Conv(n_output_chns, kernel_size=7, padding='same', kernel_initializer='he_normal', strides=1,
                     use_bias=False, name='Res1_Conv1')(input)
        conv_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_0)
        conv_0 = KL.ReLU(0.2, name='conv0_activation')(conv_0)

        conv_res = ResUnit(inshape=conv_0.shape, n_output_chns=n_output_chns,
                           kernel_size=kernel_size,
                           type_string=type_string)(conv_0)
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)
        conv_down = MaxPooling(maxpool, name='pooling')(conv_res)

        outputs=conv_down, conv_0, conv_res

        super().__init__(name='DownBlock'+i, inputs=model_inputs, outputs=outputs)
        
class ResUnit(tf.keras.Model):
    """
    adapted to TF 2 from https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/residual_unit.py
    ResUnit

        The general connections is::
            (inputs)--o-conv_0--conv_1-+-- (outputs)
                      |                |
                      o----------------o
        ``conv_0``, ``conv_1`` layers are specified by ``type_string``.

    """

    def __init__(self, input_model=None, inshape=None,
             n_output_chns=1,
             kernel_size=3,
             type_string='bn_acti_conv', i=0):
        """
               The possible types of connections are::
                   'original': residual unit presented in [2]
                   'conv_bn_acti': ReLU before addition presented in [1]
                   'acti_conv_bn': ReLU-only pre-activation presented in [1]
                   'bn_acti_conv': full pre-activation presented in [1]
                   'acti_conv': no batch-norm

               [1] recommends 'bn_acti_conv'
        """

        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            input = KL.Input(shape=inshape[1:], name='unet_input')
            model_inputs = [input]
        else:
            input = KL.concatenate(input_model.outputs, name='unet_input_concat')
            model_inputs = input_model.inputs

        ndim = 3
        conv_flow = input
        # batch normalisation layers
        bn_0 = tf.keras.layers.BatchNormalization(axis=-1)
        bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        # activation functions
        acti_0 = KL.ReLU(0.2)
        acti_1 = KL.ReLU(0.2)

        Conv = getattr(KL, 'Conv%dD' % ndim)
        conv_0 = Conv(n_output_chns, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                      strides=1,
                      use_bias=False)
        conv_1 = Conv(n_output_chns, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                      strides=1,
                      use_bias=False)

        if type_string == 'original':
            conv_flow = acti_0(bn_0(conv_0(conv_flow)))
            conv_flow = bn_1(conv_1(conv_flow))
            conv_flow = ElementwiseLayer(inshape=conv_flow.shape,inshape1=input.shape, func='SUM')([conv_flow,input])
            conv_flow = acti_1(conv_flow)
            output = conv_flow

        if type_string == 'conv_bn_acti':
            conv_flow = acti_0(bn_0(conv_0(conv_flow)))
            conv_flow = acti_1(bn_1(conv_1(conv_flow)))
            output = ElementwiseLayer(inshape=conv_flow.shape, inshape1=input.shape,
                                             func='SUM')([conv_flow, input])

        if type_string == 'acti_conv_bn':
            conv_flow = bn_0(conv_0(acti_0(conv_flow)))
            conv_flow = bn_1(conv_1(acti_1(conv_flow)))
            output = ElementwiseLayer(inshape=conv_flow.shape, inshape1=input.shape, func='SUM')([conv_flow, input])

        if type_string == 'bn_acti_conv':
            conv_flow = conv_0(acti_0(bn_0(conv_flow)))
            conv_flow = conv_1(acti_1(bn_1(conv_flow)))
            output = ElementwiseLayer(inshape=conv_flow.shape,inshape1=input.shape, func='SUM')([conv_flow,input])
        if type_string == 'acti_conv':
            conv_flow = conv_0(acti_0(conv_flow))
            conv_flow = conv_1(acti_1(conv_flow))
            output = ElementwiseLayer(inshape=conv_flow.shape, inshape1=input.shape, func='SUM')([conv_flow, input])

        super().__init__(name='ResUnit'+str(i), inputs=model_inputs, outputs=output)
        
class ElementwiseLayer(tf.keras.Model):
    """
    adapted to TF 2 from https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/elementwise.py
    """

    def __init__(self, input_model=None, inshape=None, inshape1=None,
             func='SUM', i=1):

        param_flow = KL.Input(shape=inshape[1:], name='param_flow')
        bypass_flow = KL.Input(shape=inshape1[1:], name='bypass_flow')
        model_inputs = [param_flow,bypass_flow]

        n_param_flow = param_flow.shape[-1]
        n_bypass_flow = bypass_flow.shape[-1]
        spatial_rank = ndims = 3

        output_tensor = param_flow
        if func == 'SUM':
            if n_param_flow > n_bypass_flow:  # pad the channel dim
                pad_1 = np.int((n_param_flow - n_bypass_flow) // 2)
                pad_2 = np.int(n_param_flow - n_bypass_flow - pad_1)
                padding_dims = np.vstack(([[0, 0]],
                                          [[0, 0]] * spatial_rank,
                                          [[pad_1, pad_2]]))
                bypass_flow = tf.pad(tensor=bypass_flow,
                                     paddings=padding_dims.tolist(),
                                     mode='CONSTANT')
            elif n_param_flow < n_bypass_flow:  # make a projection
                Conv = getattr(KL, 'Conv%dD' % ndims)
                projector = Conv(n_param_flow, kernel_size=1, padding='same', kernel_initializer='he_normal', strides=1)
                bypass_flow = projector(bypass_flow)

            # element-wise sum of both paths
            #output_tensor = param_flow + bypass_flow
            output_tensor =KL.add([param_flow, bypass_flow])

        elif func == 'CONCAT':
            output_tensor = tf.concat([param_flow, bypass_flow], axis=-1)

        super().__init__(name='ElementwiseLayer'+str(i), inputs=model_inputs, outputs=output_tensor)


###############################################################################
# Private functions
###############################################################################

def _conv_block(x, nfeat, strides=1, name=None, do_res=False,batch_norm=False, add_layer=None, kernel_size=3,padding='same'):
    """
    Specific convolutional block followed by leakyrelu.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal', strides=strides, name=name)(
        x)

    if batch_norm:
        convolved = tf.keras.layers.BatchNormalization(axis=-1, name=name + '_batch_norm')(convolved)

    if do_res:
        if add_layer==None:
            add_layer = x
        if nfeat != add_layer.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal',
                             name='resfix_' + name)(add_layer)
            if batch_norm:
                add_layer = tf.keras.layers.BatchNormalization(axis=-1, name='resfix_' + name + '_batch_norm')(add_layer)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])
    name = name + '_activation' if name else None

    return KL.LeakyReLU(0.2, name=name)(convolved)

def _conv_block_custom(x, nfeat, strides=1, name=None, do_res=False,batch_norm=False, add_layer=None, kernel_size=3,padding='same', activation="leakyrelu", batch_norm_mode="before_act", ndims=None):
    """
    Specific convolutional block.
    """
    if ndims ==None:
        ndims = len(x.get_shape()) - 2

    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal', strides=strides, name=name)(
        x)
    name_batch_norm = name + '_batch_norm' if name else None
    if batch_norm and batch_norm_mode=="before_act":
        convolved = tf.keras.layers.BatchNormalization(axis=-1, name=name_batch_norm)(convolved)

    if do_res:
        if add_layer==None:
            add_layer = x
        if nfeat != add_layer.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal',
                             name='resfix_' + name)(add_layer)
            if batch_norm:
                add_layer = tf.keras.layers.BatchNormalization(axis=-1, name='resfix_' + name + '_batch_norm')(add_layer)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    name_act = name + '_activation' if name else None
    if activation=="leakyrelu":
        convolved = KL.LeakyReLU(0.2, name=name_act)(convolved)
    elif activation=="relu":
        convolved = KL.ReLU(name=name_act)(convolved)
    if batch_norm and batch_norm_mode == "after_act":
        convolved = tf.keras.layers.BatchNormalization(axis=-1, name=name_batch_norm)(convolved)

    return convolved

