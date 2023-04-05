import tensorflow as tf
import tensorflow.keras.layers as KL
import math
import NetworkBasis.layers as layers

"""
    Adapted from https://github.com/cwmok/C2FViT
    Mok, T.C.W., Chung, A.C.S., 2022. Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. 
    2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 20803–20812.
"""

class Mlp_c2fvit(KL.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = KL.Dense(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer
        self.fc2 = KL.Dense(out_features)
        self.drop = KL.Dropout(drop)
        self.act2 = act_layer

    def call(self, x, H, W, D):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dwconv(x, H, W, D)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(KL.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = KL.Conv3D(dim, kernel_size = 3, padding="SAME", use_bias=False, data_format='channels_first')

    def call(self, x, H, W, D):
        B, N, C = x.shape
        if B==None:
            B=1
        x=tf.transpose(x,(0,2,1))
        x = KL.Reshape((-1, C, H, W, D))(x)
        x = self.dwconv(x)
        x = KL.Reshape((C, -1))(x)
        x = KL.Permute((2, 1))(x)
        return x

class Attention(KL.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = KL.Dense(dim, use_bias=qkv_bias)
        self.kv = KL.Dense(dim * 2, use_bias=qkv_bias)
        self.attn_drop = KL.Dropout(attn_drop)
        self.proj = KL.Dense(dim)
        self.proj_drop = KL.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = KL.Conv3D(dim, kernel_size=sr_ratio, strides=sr_ratio)
                self.norm = KL.LayerNormalization()
        else:
            self.pool = AdaptiveAvgPool3D(7)
            self.sr = KL.Conv3D(dim, kernel_size=1, strides=1)
            self.norm = KL.LayerNorm(dim)
            self.act = tf.nn.gelu()

    def call(self, x, H, W, D):
        B, N, C = x.shape
        q=self.q(x)
        if B==None:
            B=1
        q=KL.Reshape((B, N, self.num_heads, C // self.num_heads))(q)
        q=KL.Permute((1, 3, 2, 4))(q)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = tf.reshape(tf.transpose(x, perm=(0, 2, 1)), shape=(B, C, H, W, D))
                x_ = self.sr(x_)
                x_ = tf.reshape(x_, shape=(B, C, -1))
                x_ = tf.transpose(x_, perm=(0, 2, 1))
                x_ = self.norm(x_)
                kv = self.kv(x_)
                kv = tf.reshape(kv, shape=(B, -1, 2, self.num_heads, C // self.num_heads))
                kv = tf.transpose(kv, perm=(2, 0, 3, 1, 4))
            else:
                kv = self.kv(x)
                kv = KL.Reshape((B, -1, 2, self.num_heads, C // self.num_heads))(kv)
                kv = KL.Permute((3, 1, 4, 2, 5))(kv)
        else:
            x_ = tf.reshape(tf.transpose(x, perm=(0, 2, 1)), shape=(B, C, H, W, D))
            x_ = self.sr(self.pool(x_))
            x_ = tf.reshape(x_, shape=(B, C, -1))
            x_ = tf.transpose(x_, perm=(0, 2, 1))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_)
            kv= tf.reshape(kv,B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[:,0,:,:,:,:], kv[:,1,:,:,:,:]

        attn = (q @ KL.Permute((1,2,4,3))(k))* self.scale
        attn=KL.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = KL.Permute((2, 1, 3, 4))(x)
        x = KL.Reshape((N, C))(x)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AdaptiveAvgPool3D(KL.Layer):
    '''
    Implementation of AdaptiveAvgPool3D as used in pyTorch impl.
    https://github.com/Chianugoogidi/X3D-tf/blob/main/model.py
    '''

    def __init__(self,
                 spatial_out_shape=(1, 1, 1),
                 data_format='channels_first',
                 **kwargs) -> None:
        super(AdaptiveAvgPool3D, self).__init__(**kwargs)
        assert len(spatial_out_shape) == 3, "Please specify 3D shape"
        assert data_format in ('channels_last', 'channels_first')

        self.data_format = data_format
        self.out_shape = spatial_out_shape
        self.avg_pool = KL.GlobalAveragePooling3D()

    def call(self, input):
        out = self.avg_pool(input)
        if self.data_format == 'channels_last':
            return tf.reshape(
                out,
                shape=(
                    -1,
                    self.out_shape[0],
                    self.out_shape[1],
                    self.out_shape[2],
                    out.shape[1]))
        else:
            return tf.reshape(
                out,
                shape=(
                    -1,
                    out.shape[1],
                    self.out_shape[0],
                    self.out_shape[1],
                    self.out_shape[2]))

class Block(KL.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=KL.LayerNormalization, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_c2fvit(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def call(self, x, H, W, D):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))
        return x

class OverlapPatchEmbed(KL.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=3, embed_dim=768, flatten=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W, self.D = img_size[0] // stride, img_size[1] // stride, img_size[2] // stride
        self.num_patches = self.H * self.W * self.D
        self.proj = KL.Conv3D(embed_dim, kernel_size=patch_size, strides=stride,padding="SAME", data_format="channels_first" )
        # self.norm = nn.LayerNorm(embed_dim)
        self.flatten = flatten

        self.act = tf.nn.gelu

    def call(self, x):
        x = self.proj(x)
        _, C, H, W, D = x.shape
        if self.flatten:
            # BCHW -> BNC
            x = KL.Flatten()(x)
            x = KL.Reshape((C, H*W*D))(x)
            x = tf.transpose(x, [0,2,1])
        # x = self.norm(x)
        x = self.act(x)
        return x, H, W, D

class AffineCOMTransform(KL.Layer):
    """
    Computes the corresponding (flattened) affine from a vector of transform
    components. The components are in the order of (translation, rotation, shearing, scaling), so the
    input must a 1D array of length (ndim * 4).
    """

    def __init__(self, use_com=True):
        self.translation_m = tf.Variable(tf.eye(4), dtype=tf.float32, name="translation_m")
        self.rotation_x = tf.Variable(tf.eye(4), dtype=tf.float32, name="rotation_x")
        self.rotation_y = tf.Variable(tf.eye(4), dtype=tf.float32, name="rotation_y")
        self.rotation_z = tf.Variable(tf.eye(4), dtype=tf.float32, name="rotation_z")
        self.rotation_m = tf.Variable(tf.eye(4), dtype=tf.float32, name="rotation_m")
        self.shearing_m = tf.Variable(tf.eye(4), dtype=tf.float32, name="shearing_m")
        self.scaling_m = tf.Variable(tf.eye(4), dtype=tf.float32, name="scaling_m")

        self.to_center_matrix = tf.Variable(tf.eye(4), dtype=tf.float32, name="to_center_matrix")
        self.reversed_to_center_matrix = tf.Variable(tf.eye(4), dtype=tf.float32, name="reversed_to_center_matrix")

        self.id = tf.Variable(tf.zeros((1, 3, 4), name="id"))
        self.id[0, 0, 0].assign(1)
        self.id[0, 1, 1].assign(1)
        self.id[0, 2, 2].assign(1)

        self.use_com = use_com
        super().__init__()

    def call(self, x, affine_para):
        # Matrix that register x to its center of mass
        #Used affine_to_shift (VoxelMorph) as no function like F.affine_grid is available in tensorflow
        inshape=x.shape[2:]
        aff = lambda x: layers.affine_to_shift(x, inshape, shift_center=True)
        id_grid = tf.map_fn(aff, KL.Flatten()(KL.Flatten()(self.id)), dtype=tf.float32)

        if self.use_com:
            x_sum = tf.reduce_sum(x)
            center_mass_x = tf.reduce_sum(KL.Permute((2, 3, 4, 1))(x)[..., 0] * id_grid[..., 0]) / x_sum
            center_mass_y = tf.reduce_sum(KL.Permute((2, 3, 4, 1))(x)[..., 0] * id_grid[..., 1]) / x_sum
            center_mass_z = tf.reduce_sum(KL.Permute((2, 3, 4, 1))(x)[..., 0] * id_grid[..., 2]) / x_sum

            self.to_center_matrix[0, 3].assign(center_mass_x)
            self.to_center_matrix[1, 3].assign(center_mass_y)
            self.to_center_matrix[2, 3].assign(center_mass_z)
            self.reversed_to_center_matrix[0, 3].assign(-center_mass_x)
            self.reversed_to_center_matrix[1, 3].assign(-center_mass_y)
            self.reversed_to_center_matrix[2, 3].assign(-center_mass_z)

        trans_xyz = affine_para[0, 0:3]
        rotate_xyz = affine_para[0, 3:6] * math.pi
        shearing_xyz = affine_para[0, 6:9] * math.pi
        scaling_xyz = 1 + (affine_para[0, 9:12] * 0.5)

        self.translation_m[0, 3].assign(trans_xyz[0])
        self.translation_m[1, 3].assign(trans_xyz[1])
        self.translation_m[2, 3].assign(trans_xyz[2])
        self.scaling_m[0, 0].assign(scaling_xyz[0])
        self.scaling_m[1, 1].assign(scaling_xyz[1])
        self.scaling_m[2, 2].assign(scaling_xyz[2])

        self.rotation_x[1, 1].assign(tf.math.cos(rotate_xyz[0]))
        self.rotation_x[1, 2].assign(-tf.math.sin(rotate_xyz[0]))
        self.rotation_x[2, 1].assign(tf.math.sin(rotate_xyz[0]))
        self.rotation_x[2, 2].assign(tf.math.cos(rotate_xyz[0]))

        self.rotation_y[0, 0].assign(tf.math.cos(rotate_xyz[1]))
        self.rotation_y[0, 2].assign(tf.math.sin(rotate_xyz[1]))
        self.rotation_y[2, 0].assign(-tf.math.sin(rotate_xyz[1]))
        self.rotation_y[2, 2].assign(tf.math.cos(rotate_xyz[1]))

        self.rotation_z[0, 0].assign(tf.math.cos(rotate_xyz[2]))
        self.rotation_z[0, 2].assign(-tf.math.sin(rotate_xyz[2]))
        self.rotation_z[2, 0].assign(tf.math.sin(rotate_xyz[2]))
        self.rotation_z[2, 2].assign(tf.math.cos(rotate_xyz[2]))

        self.rotation_m = tf.math.multiply(tf.math.multiply(self.rotation_z, self.rotation_y), self.rotation_x)

        self.shearing_m[0, 1].assign(shearing_xyz[0])
        self.shearing_m[0, 2].assign(shearing_xyz[1])
        self.shearing_m[1, 2].assign(shearing_xyz[2])

        output_affine_m = tf.math.multiply(self.to_center_matrix,tf.math.multiply(self.shearing_m,tf.math.multiply
            (self.scaling_m, tf.math.multiply(self.rotation_m,tf.math.multiply(self.reversed_to_center_matrix,self.translation_m)))))

        return tf.reshape(output_affine_m[:,:3], (1,12))