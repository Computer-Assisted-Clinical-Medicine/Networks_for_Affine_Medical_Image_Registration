import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as KL
import neurite as ne
import math

class SpatialTransformer_with_disp_swin(Layer):
    """
    Adapted from Voxelmorph: https://github.com/voxelmorph/voxelmorph
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix (unless specified otherwise).

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 add_identity=True,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.
            add_identity (default: True): whether the identity matrix is added
                to affine transforms.
            shift_center (default: True): whether the grid is shifted to the center
                of the image when converting affine transforms to warp fields.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.add_identity = add_identity
        self.shift_center = shift_center
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'add_identity': self.add_identity,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1] or [N, N+1]
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        is_matrix = len(trf_shape) == 2 and trf_shape[0] in (self.ndims, self.ndims + 1) and trf_shape[
            1] == self.ndims + 1
        self.is_affine = len(trf_shape) == 1 or is_matrix
        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            print("affine")
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d'
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]
        trf = inputs[1][0]
        #print("trf",trf)

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # convert matrix to warp field
        if self.is_affine:
            ncols = self.ndims + 1
            nrows = self.ndims
            if np.prod(trf.shape.as_list()[1:]) == (self.ndims + 1) ** 2:
                nrows += 1
            if len(trf.shape[1:]) == 1:
                trf = tf.reshape(trf, shape=(-1, nrows, ncols))
            if self.add_identity:
                trf += tf.eye(nrows, ncols, batch_shape=(tf.shape(trf)[0],))
            # print("trf", trf.shape)
            fun = lambda x: affine_to_shift(x, vol.shape[1:-1], shift_center=self.shift_center)
            trf = tf.map_fn(fun, trf, dtype=tf.float32)
            # print("trf", trf.shape)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0, :]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32), trf

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

class SpatialTransformer_with_disp(Layer):
    """
    Adapted from Voxelmorph: https://github.com/voxelmorph/voxelmorph
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix (unless specified otherwise).

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 add_identity=True,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.
            add_identity (default: True): whether the identity matrix is added
                to affine transforms.
            shift_center (default: True): whether the grid is shifted to the center
                of the image when converting affine transforms to warp fields.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.add_identity = add_identity
        self.shift_center = shift_center
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'add_identity': self.add_identity,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        #print(trf_shape)

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1] or [N, N+1]
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        is_matrix = len(trf_shape) == 2 and trf_shape[0] in (self.ndims, self.ndims + 1) and trf_shape[
            1] == self.ndims + 1
        self.is_affine = len(trf_shape) == 1 or is_matrix

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            print("affine")
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d'
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # convert matrix to warp field
        if self.is_affine:
            ncols = self.ndims + 1
            nrows = self.ndims
            if np.prod(trf.shape.as_list()[1:]) == (self.ndims + 1) ** 2:
                nrows += 1
            if len(trf.shape[1:]) == 1:
                trf = tf.reshape(trf, shape=(-1, nrows, ncols))
            if self.add_identity:
                trf += tf.eye(nrows, ncols, batch_shape=(tf.shape(trf)[0],))
            #print("trf", trf.shape)
            fun = lambda x: affine_to_shift(x, vol.shape[1:-1], shift_center=self.shift_center)
            trf = tf.map_fn(fun, trf, dtype=tf.float32)
            #print("trf", trf.shape)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0, :]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32), trf

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    Adapted from Voxelmorph: https://github.com/voxelmorph/voxelmorph
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    TODO:
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    if isinstance(volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()

    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1) (ndims=%d).'
                             'Got len %d' % (nb_dims, len(affine_matrix)))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        shape1 = '(%d x %d)' % (nb_dims + 1, nb_dims + 1)
        shape2 = '(%d x %s)' % (nb_dims, nb_dims + 1)
        true_shape = str(affine_matrix.shape)
        raise Exception('Affine shape should match %s or %s, but got: %s' % (shape1, shape2, true_shape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    Adapted from Voxelmorph: https://github.com/voxelmorph/voxelmorph
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes

    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)

class FeatureL2NormLayer(KL.Layer):
    def __init__(self):
        super(FeatureL2NormLayer, self).__init__()

    def call(self, inputs):
        epsilon = 1e-6
        norm=tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(inputs,2),1)+epsilon)
        return tf.divide(inputs,norm)

class FeatureCorrelation(KL.Layer):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def call(self, feature_A, feature_B):
        b, h, w, s, c = feature_A.shape
        # reshape features for matrix multiplication
        feature_A = tf.transpose(feature_A, [0, 4, 3, 2, 1]) # b, c, s, w, h
        feature_A = tf.reshape(feature_A, [-1, c, h * w * s])
        feature_B = tf.reshape(feature_B, [-1, h * w * s, c])
        # perform matrix mult.
        feature_mul = tf.matmul(feature_B, feature_A)
        correlation_tensor = tf.reshape(feature_mul, [-1, h, w, s, h * w * s])
        return correlation_tensor

class FeatureRegression(KL.Layer):
    def __init__(self, output_dim=12): #12 instead of 6 because 3D images are used
        self.conv = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, kernel_size=7, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv3D(64, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
        ])
        self.linear = tf.keras.layers.Dense(output_dim)
        super(FeatureRegression, self).__init__()

    def call(self, x):
        x = self.conv(x)
        x = KL.Flatten()(x)
        x = self.linear(x)
        return x

class AffineTransformationsToMatrix(Layer):
    """
    Computes the corresponding (flattened) affine from a vector of transform
    components. The components are in the order of (translation, rotation, shearing, scaling), so the
    input must a 1D array of length (ndim * 4).
    """

    def __init__(self, ndims, use_constraint=False, **kwargs):
        self.ndims = ndims
        self.use_constraint = use_constraint
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('rigid registration is limited to 3D for now')
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, vector):
        """
        Parameters
            vector: tensor of affine components
            translation,rotation,shearing,scaling
        """
        return tf.map_fn(self._single_conversion, vector, dtype=tf.float32)

    def _single_conversion(self, vector):

        if self.ndims == 3:
            if self.use_constraint:
                # clip rotation, shearing and scaling values
                vector = tf.concat([
                    vector[:3],
                    tf.clip_by_value(vector[3:6], -math.pi, math.pi),
                    tf.clip_by_value(vector[6:9], -math.pi, math.pi),
                    tf.clip_by_value(vector[9:12], 0.5, 1.5)
                ], axis=0)
            # extract components of input vector
            translation = vector[:3]
            angle_x = vector[3]
            angle_y = vector[4]
            angle_z = vector[5]
            shearing= vector[6:9]
            scaling = vector[9:]

            # x rotation matrix
            cosx = tf.math.cos(angle_x)
            sinx = tf.math.sin(angle_x)
            x_rot = tf.convert_to_tensor([
                [1, 0, 0],
                [0, cosx, -sinx],
                [0, sinx, cosx]
            ], name='x_rot')

            # y rotation matrix
            cosy = tf.math.cos(angle_y)
            siny = tf.math.sin(angle_y)
            y_rot = tf.convert_to_tensor([
                [cosy, 0, siny],
                [0, 1, 0],
                [-siny, 0, cosy]
            ], name='y_rot')

            # z rotation matrix
            cosz = tf.math.cos(angle_z)
            sinz = tf.math.sin(angle_z)
            z_rot = tf.convert_to_tensor([
                [cosz, -sinz, 0],
                [sinz, cosz, 0],
                [0, 0, 1]
            ], name='z_rot')

            # compose matrices
            t_rot = tf.tensordot(x_rot, y_rot, 1)
            m_rot = tf.tensordot(t_rot, z_rot, 1)

            # build scale matrix
            m_scale = tf.convert_to_tensor([
                [scaling[0], 0, 0],
                [0, scaling[1], 0],
                [0, 0, scaling[2]]
            ], name='scale')

            m_shear = tf.convert_to_tensor([
                [0, shearing[0], shearing[1]],
                [0,1, shearing[2]],
                [0, 0, 1]
            ], name='scale')

        elif self.ndims == 2:
            # extract components of input vector
            translation = vector[:2]
            angle = vector[2]

            # rotation matrix
            cosz = tf.math.cos(angle)
            sinz = tf.math.sin(angle)
            m_rot = tf.convert_to_tensor([
                [cosz, -sinz],
                [sinz, cosz]
            ], name='rot')

            s = vector[3] if self.scale else 1.0
            m_scale = tf.convert_to_tensor([
                [s, 0],
                [0, s]
            ], name='scale')

        # we want to encode shift transforms, so remove identity
        m_rot -= tf.eye(self.ndims)

        # scale the matrix
        m_rot = tf.tensordot(m_rot, m_scale, 1)

        # shear the matrix
        m_rot = tf.tensordot(m_rot, m_shear, 1)

        # concat the linear translation
        matrix = tf.concat([m_rot, tf.expand_dims(translation, 1)], 1)

        # flatten
        affine = tf.reshape(matrix, [self.ndims * (self.ndims + 1)])
        return affine

class AffineTransformationsToMatrix_swin(Layer):
    """
    3-D Affine Transformer
    """

    def __init__(self, mode='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, src, affine, scale, translate, shear):
        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        theta_z = affine[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = tf.stack([tf.stack([tf.ones_like(theta_x), tf.zeros_like(theta_x), tf.zeros_like(theta_x)], axis=1), tf.stack([tf.zeros_like(theta_x), tf.cos(theta_x), -tf.sin(theta_x)], axis=1), tf.stack([tf.zeros_like(theta_x), tf.sin(theta_x), tf.cos(theta_x)], axis=1)], axis=2)
        rot_mat_y = tf.stack([tf.stack([tf.cos(theta_y), tf.zeros_like(theta_y), tf.sin(theta_y)], axis=1), tf.stack([tf.zeros_like(theta_y), tf.ones_like(theta_x), tf.zeros_like(theta_x)], axis=1), tf.stack([-tf.sin(theta_y), tf.zeros_like(theta_y), tf.cos(theta_y)], axis=1)], axis=2)
        rot_mat_z = tf.stack([tf.stack([tf.cos(theta_z), -tf.sin(theta_z), tf.zeros_like(theta_y)], axis=1), tf.stack([tf.sin(theta_z), tf.cos(theta_z), tf.zeros_like(theta_y)], axis=1), tf.stack([tf.zeros_like(theta_y), tf.zeros_like(theta_y), tf.ones_like(theta_x)], axis=1)], axis=2)
        scale_mat = tf.stack(
            [tf.stack([scale_x, tf.zeros_like(theta_z), tf.zeros_like(theta_y)], axis=1),
             tf.stack([tf.zeros_like(theta_z), scale_y, tf.zeros_like(theta_y)], axis=1),
             tf.stack([tf.zeros_like(theta_y), tf.zeros_like(theta_y), scale_z], axis=1)], axis=2)
        shear_mat = tf.stack(
            [tf.stack([tf.ones_like(theta_x), tf.tan(shear_xy), tf.tan(shear_xz)], axis=1),
             tf.stack([tf.tan(shear_yx), tf.ones_like(theta_x), tf.tan(shear_yz)], axis=1),
             tf.stack([tf.tan(shear_zx), tf.tan(shear_zy), tf.ones_like(theta_x)], axis=1)], axis=2)
        trans = tf.stack([trans_x, trans_y, trans_z], axis=1)[:, :, tf.newaxis]#[:, tf.newaxis, :]
        mat = tf.linalg.matmul(rot_mat_x, rot_mat_y)
        mat = tf.linalg.matmul(rot_mat_z, mat)
        mat = tf.linalg.matmul(scale_mat, mat)
        mat = tf.linalg.matmul(shear_mat, mat)

        mat = tf.concat([mat, trans], axis=-1)

        return mat

def clamp(n, minn, maxn):
    return tf.math.maximum(tf.math.minimum(maxn, n), minn)

def Combining_Affine_Para3D(tensors):
    """
    https://github.com/xuuuuuuchen/PASTA/
    """
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]

    tx = tf.squeeze(tf.slice(array, [0, 0], [n_batch, 1]), 1)
    ty = tf.squeeze(tf.slice(array, [0, 1], [n_batch, 1]), 1)
    tz = tf.squeeze(tf.slice(array, [0, 2], [n_batch, 1]), 1)

    sin0x = tf.squeeze(tf.slice(array, [0, 3], [n_batch, 1]), 1)
    sin0y = tf.squeeze(tf.slice(array, [0, 4], [n_batch, 1]), 1)
    sin0z = tf.squeeze(tf.slice(array, [0, 5], [n_batch, 1]), 1)
    cos0x = tf.sqrt(1.0 - tf.square(sin0x))
    cos0y = tf.sqrt(1.0 - tf.square(sin0y))
    cos0z = tf.sqrt(1.0 - tf.square(sin0z))

    shxy = tf.squeeze(tf.slice(array, [0, 6], [n_batch, 1]), 1)
    shyx = tf.squeeze(tf.slice(array, [0, 7], [n_batch, 1]), 1)
    shxz = tf.squeeze(tf.slice(array, [0, 8], [n_batch, 1]), 1)
    shzx = tf.squeeze(tf.slice(array, [0, 9], [n_batch, 1]), 1)
    shzy = tf.squeeze(tf.slice(array, [0, 10], [n_batch, 1]), 1)
    shyz = tf.squeeze(tf.slice(array, [0, 11], [n_batch, 1]), 1)

    scx = tf.squeeze(tf.slice(array, [0, 12], [n_batch, 1]), 1)
    scy = tf.squeeze(tf.slice(array, [0, 13], [n_batch, 1]), 1)
    scz = tf.squeeze(tf.slice(array, [0, 14], [n_batch, 1]), 1)
    """
    CORE
    """
    x1 = -shxz * scx * sin0y + scx * cos0y * cos0z + shxy * scx * cos0y * sin0z

    x2 = shxz * scx * sin0x * cos0y + shxy * scx * cos0x * cos0z + scx * sin0x * sin0y * cos0z - scx * cos0x * sin0z + shxy * scx * sin0x * sin0y * sin0z

    x3 = shxz * scx * cos0x * cos0y - shxy * scx * sin0x * cos0z + scx * cos0x * sin0y * cos0z + scx * sin0x * sin0z + shxy * scx * cos0x * sin0y * sin0z

    x4 = scy * cos0y * sin0z + scy * cos0y * cos0z * shyx - scy * sin0y * shyz

    x5 = scy * cos0x * cos0z + scy * sin0x * sin0y * sin0z + scy * sin0x * sin0y * cos0z * shyx - scy * cos0x * sin0z * shyx + scy * sin0x * cos0y * shyz

    x6 = -scy * sin0x * cos0z + scy * cos0x * sin0y * sin0z + scy * cos0x * sin0y * cos0z * shyx + scy * sin0x * sin0z * shyx + scy * cos0x * cos0y * shyz

    x7 = -scz * sin0y + shzx * scz * cos0y * cos0z + shzy * scz * cos0y * sin0z

    x8 = scz * sin0x * cos0y + shzy * scz * cos0x * cos0z + shzx * scz * sin0x * sin0y * cos0z - shzx * scz * cos0x * sin0z + shzy * scz * sin0x * sin0y * sin0z

    x9 = scz * cos0x * cos0y - shzy * scz * sin0x * cos0z + shzx * scz * cos0x * sin0y * cos0z + shzx * scz * sin0x * sin0z + shzy * scz * cos0x * sin0y * sin0z

    x10 = tx
    x11 = ty
    x12 = tz

    x1 = tf.expand_dims(x1, 1)
    x2 = tf.expand_dims(x2, 1)
    x3 = tf.expand_dims(x3, 1)
    x4 = tf.expand_dims(x4, 1)
    x5 = tf.expand_dims(x5, 1)
    x6 = tf.expand_dims(x6, 1)
    x7 = tf.expand_dims(x7, 1)
    x8 = tf.expand_dims(x8, 1)
    x9 = tf.expand_dims(x9, 1)
    x10 = tf.expand_dims(x10, 1)
    x11 = tf.expand_dims(x11, 1)
    x12 = tf.expand_dims(x12, 1)

    array = tf.concat([x1, x2, x3, x10,
                       x4, x5, x6, x11,
                       x7, x8, x9, x12, ], 1)

    # print(" >>>>>>>>>> matrix: "+str(array.shape))
    return array