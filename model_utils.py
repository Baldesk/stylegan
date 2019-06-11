from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Dropout, LeakyReLU, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, concatenate


from tensorflow.keras.layers import MaxPooling2D

import facenet_utils
from facenet_utils import LRN2D

def Leaky_ResNet_block(input_tensor, filters, strides=1):
    kernel_size = 3
    first_padding = 'same'
    if strides == 2:
        first_padding = 'valid'
    #x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    #print(x)
    #x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    shortcut = input_tensor
    if strides == 2:
        #shortcut = Conv2D(filters, (1,1), strides=strides, kernel_regularizer=regularizers.l2(0.01))(shortcut)
        shortcut = Conv2D(filters, (1, 1), strides=strides)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])

    x = LeakyReLU()(x)
    #print(x)
    return x


def ResNet_block(input_tensor, filters, strides=1):
    kernel_size = 3
    first_padding = 'same'
    if strides == 2:
        first_padding = 'valid'
    #x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    #print(x)
    #x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    shortcut = input_tensor
    if strides == 2:
        #shortcut = Conv2D(filters, (1,1), strides=strides, kernel_regularizer=regularizers.l2(0.01))(shortcut)
        shortcut = Conv2D(filters, (1, 1), strides=strides)(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])

    x = Activation('relu')(x)
    #print(x)
    return x

#################################################################
# ResNet Architecture               | Output Shape
# ----------------------------------|---------------
# 1. Conv 7x7 - 64                  | 96,96,64
# 2. MaxPool /2                     | 48,48,64
# 3. ConvBlock - 64                 | 48,48,64
# 4. ConvBlock - 64                 | 48,48,64
# 5. ConvBlock - 64                 | 48,48,64
# 6. ConvBlock - 128 WITH STRIDE=2  | 24,24,128
# 7. ConvBlock - 128                | 24,24,128
# 8. ConvBlock - 128                | 24,24,128
# 9. ConvBlock - 128                | 24,24,128
# 10. ConvBlock - 256 WITH STRIDE=2 | 12,12,256
# 11. ConvBlock - 256               | 12,12,256
# 12. ConvBlock - 256               | 12,12,256
# 13. ConvBlock - 256               | 12,12,256
# 14. ConvBlock - 256               | 12,12,256
# 15. ConvBlock - 256               | 12,12,256
# 16. ConvBlock - 512 WITH STRIDE=2 | 6,6,512
# 17. ConvBlock - 512               | 6,6,512
# 18. ConvBlock - 512               | 6,6,512
# 19. Global AveragePool            | 6,6,512
# 20. Output Softmax Layer          | Number of classes Used
#################################################################
def ResNet_96(num_classes, top=True):
    input_shape = (96,96,3)
    img_input = Input(input_shape)
    #x = Conv2D(64, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = Conv2D(64, 7, padding='same')(img_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ResNet_block(x, 64)
    x = ResNet_block(x,64)
    x = ResNet_block(x,64)

    x = ResNet_block(x,128, 2)
    x = ResNet_block(x, 128)
    x = ResNet_block(x, 128)
    x = ResNet_block(x, 128)

    x = ResNet_block(x, 256, 2)
    x = ResNet_block(x, 256)
    x = ResNet_block(x, 256)
    x = ResNet_block(x, 256)
    x = ResNet_block(x, 256)
    x = ResNet_block(x, 256)

    x = ResNet_block(x, 512, 2)
    x = ResNet_block(x, 512)
    x = ResNet_block(x, 512)
    if(not top):
        model = Model(inputs=img_input, outputs=x)
        return model
    x = AveragePooling2D((6,6))(x)
    x = Flatten()(x)

   ####
    #x = Dense(1024)(x)
    #x = Dropout(0.3)(x)
    #x = Activation('relu')(x)
   ####

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model


def Leaky_ResNet_96(num_classes, top=True):
    input_shape = (96, 96, 3)
    img_input = Input(input_shape, name='LeakyResNet96_input')
    # x = Conv2D(64, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = Conv2D(64, 7, padding='same')(img_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Leaky_ResNet_block(x, 64)
    x = Leaky_ResNet_block(x, 64)
    x = Leaky_ResNet_block(x, 64)

    x = Leaky_ResNet_block(x, 128, 2)
    x = Leaky_ResNet_block(x, 128)
    x = Leaky_ResNet_block(x, 128)
    x = Leaky_ResNet_block(x, 128)

    x = Leaky_ResNet_block(x, 256, 2)
    x = Leaky_ResNet_block(x, 256)
    x = Leaky_ResNet_block(x, 256)
    x = Leaky_ResNet_block(x, 256)
    x = Leaky_ResNet_block(x, 256)
    x = Leaky_ResNet_block(x, 256)

    x = Leaky_ResNet_block(x, 512, 2)
    x = Leaky_ResNet_block(x, 512)
    x = Leaky_ResNet_block(x, 512)

    x = AveragePooling2D((6, 6))(x)
    if (not top):
        model = Model(inputs=img_input, outputs=x)
        return model
    x = Flatten()(x)

    ####
    # x = Dense(1024)(x)
    # x = Dropout(0.3)(x)
    # x = Activation('relu')(x)
    ####

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model





def ResNet_224(num_classes):
    input_shape = (224,224,3)
    img_input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(img_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ResNet_block(x, 64)
    x = ResNet_block(x,64)
    #x = ResNet_block(x,64)

    x = ResNet_block(x,128, 2)
    x = ResNet_block(x, 128)
    #x = ResNet_block(x, 128)
    #x = ResNet_block(x, 128)

    x = ResNet_block(x, 256, 2)
    x = ResNet_block(x, 256)
    x = ResNet_block(x, 256)
    #x = ResNet_block(x, 256)
    #x = ResNet_block(x, 256)
    #x = ResNet_block(x, 256)

    x = ResNet_block(x, 512, 2)
    x = ResNet_block(x, 512)
    #x = ResNet_block(x, 512)

    x = AveragePooling2D((6,6))(x)
    x = Flatten()(x)

   # ####
    #x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Flatten()(x)
   # ####

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model




# Do not use batch norm assuming input shape 96,96,3
def custom_cnn(input_shape = (96, 96, 3)):
    img_input = Input(input_shape)
    # x = Conv2D(64, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = Conv2D(64, 3, padding='same')(img_input)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(48x48)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # (24x24)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # (12x12)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Conv2D(512, 3, padding='same')(x)
    #x = LeakyReLU()(x)
    #x = Conv2D(512, 3, padding='same')(x)
    #x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # (6x6x512)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Conv2D(512, 3, padding='same')(x)
    #x = LeakyReLU()(x)
    #x = Conv2D(512, 3, padding='same')(x)
    #x = LeakyReLU()(x)

    # x = ResNet_block(x, 64)
    # x = ResNet_block(x, 64)
    # x = ResNet_block(x, 64)
    #
    # x = ResNet_block(x, 128, 2)
    # x = ResNet_block(x, 128)
    # x = ResNet_block(x, 128)
    # x = ResNet_block(x, 128)
    #
    # x = ResNet_block(x, 256, 2)
    # x = ResNet_block(x, 256)
    # x = ResNet_block(x, 256)
    # x = ResNet_block(x, 256)
    # x = ResNet_block(x, 256)
    # x = ResNet_block(x, 256)
    #
    # x = ResNet_block(x, 512, 2)
    # x = ResNet_block(x, 512)
    # x = ResNet_block(x, 512)
    model = Model(inputs=img_input, outputs=x)
    return model




def FaceNet1024(include_norm=False):
    myInput = Input(shape=(1024, 1024, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(1024, 1024, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3a)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = facenet_utils.conv2d_bn(inception_3b,
                                               layer='inception_3c_3x3',
                                               cv1_out=128,
                                               cv1_filter=(1, 1),
                                               cv2_out=256,
                                               cv2_filter=(3, 3),
                                               cv2_strides=(2, 2),
                                               padding=(1, 1))

    inception_3c_5x5 = facenet_utils.conv2d_bn(inception_3b,
                                               layer='inception_3c_5x5',
                                               cv1_out=32,
                                               cv1_filter=(1, 1),
                                               cv2_out=64,
                                               cv2_filter=(5, 5),
                                               cv2_strides=(2, 2),
                                               padding=(2, 2))

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    # inception 4a
    inception_4a_3x3 = facenet_utils.conv2d_bn(inception_3c,
                                               layer='inception_4a_3x3',
                                               cv1_out=96,
                                               cv1_filter=(1, 1),
                                               cv2_out=192,
                                               cv2_filter=(3, 3),
                                               cv2_strides=(1, 1),
                                               padding=(1, 1))
    inception_4a_5x5 = facenet_utils.conv2d_bn(inception_3c,
                                               layer='inception_4a_5x5',
                                               cv1_out=32,
                                               cv1_filter=(1, 1),
                                               cv2_out=64,
                                               cv2_filter=(5, 5),
                                               cv2_strides=(1, 1),
                                               padding=(2, 2))

    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3c)
    inception_4a_pool = facenet_utils.conv2d_bn(inception_4a_pool,
                                                layer='inception_4a_pool',
                                                cv1_out=128,
                                                cv1_filter=(1, 1),
                                                padding=(2, 2))
    inception_4a_1x1 = facenet_utils.conv2d_bn(inception_3c,
                                               layer='inception_4a_1x1',
                                               cv1_out=256,
                                               cv1_filter=(1, 1))
    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    # inception4e
    inception_4e_3x3 = facenet_utils.conv2d_bn(inception_4a,
                                               layer='inception_4e_3x3',
                                               cv1_out=160,
                                               cv1_filter=(1, 1),
                                               cv2_out=256,
                                               cv2_filter=(3, 3),
                                               cv2_strides=(2, 2),
                                               padding=(1, 1))
    inception_4e_5x5 = facenet_utils.conv2d_bn(inception_4a,
                                               layer='inception_4e_5x5',
                                               cv1_out=64,
                                               cv1_filter=(1, 1),
                                               cv2_out=128,
                                               cv2_filter=(5, 5),
                                               cv2_strides=(2, 2),
                                               padding=(2, 2))
    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

    # inception5a
    inception_5a_3x3 = facenet_utils.conv2d_bn(inception_4e,
                                               layer='inception_5a_3x3',
                                               cv1_out=96,
                                               cv1_filter=(1, 1),
                                               cv2_out=384,
                                               cv2_filter=(3, 3),
                                               cv2_strides=(1, 1),
                                               padding=(1, 1))

    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4e)
    inception_5a_pool = facenet_utils.conv2d_bn(inception_5a_pool,
                                                layer='inception_5a_pool',
                                                cv1_out=96,
                                                cv1_filter=(1, 1),
                                                padding=(1, 1))
    inception_5a_1x1 = facenet_utils.conv2d_bn(inception_4e,
                                               layer='inception_5a_1x1',
                                               cv1_out=256,
                                               cv1_filter=(1, 1))

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    # inception_5b
    inception_5b_3x3 = facenet_utils.conv2d_bn(inception_5a,
                                               layer='inception_5b_3x3',
                                               cv1_out=96,
                                               cv1_filter=(1, 1),
                                               cv2_out=384,
                                               cv2_filter=(3, 3),
                                               cv2_strides=(1, 1),
                                               padding=(1, 1))
    inception_5b_3x3_maxpool = MaxPooling2D(inception_5b_3x3)
    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = facenet_utils.conv2d_bn(inception_5b_pool,
                                                layer='inception_5b_pool',
                                                cv1_out=96,
                                                cv1_filter=(1, 1))
    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = facenet_utils.conv2d_bn(inception_5a,
                                               layer='inception_5b_1x1',
                                               cv1_out=256,
                                               cv1_filter=(1, 1))
    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    if include_norm:
        norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)
        return Model(inputs=[myInput], outputs=norm_layer)
    return Model(inputs=[myInput], outputs=dense_layer)









def FaceNet96(include_norm=False):

    myInput = Input(shape=(96, 96, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3a)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = facenet_utils.conv2d_bn(inception_3b,
                                       layer='inception_3c_3x3',
                                       cv1_out=128,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))

    inception_3c_5x5 = facenet_utils.conv2d_bn(inception_3b,
                                       layer='inception_3c_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    #inception 4a
    inception_4a_3x3 = facenet_utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=192,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    inception_4a_5x5 = facenet_utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(1, 1),
                                       padding=(2, 2))

    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3c)
    inception_4a_pool = facenet_utils.conv2d_bn(inception_4a_pool,
                                        layer='inception_4a_pool',
                                        cv1_out=128,
                                        cv1_filter=(1, 1),
                                        padding=(2, 2))
    inception_4a_1x1 = facenet_utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    #inception4e
    inception_4e_3x3 = facenet_utils.conv2d_bn(inception_4a,
                                       layer='inception_4e_3x3',
                                       cv1_out=160,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))
    inception_4e_5x5 = facenet_utils.conv2d_bn(inception_4a,
                                       layer='inception_4e_5x5',
                                       cv1_out=64,
                                       cv1_filter=(1, 1),
                                       cv2_out=128,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))
    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

    #inception5a
    inception_5a_3x3 = facenet_utils.conv2d_bn(inception_4e,
                                       layer='inception_5a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))

    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4e)
    inception_5a_pool = facenet_utils.conv2d_bn(inception_5a_pool,
                                        layer='inception_5a_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1),
                                        padding=(1, 1))
    inception_5a_1x1 = facenet_utils.conv2d_bn(inception_4e,
                                       layer='inception_5a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    #inception_5b
    inception_5b_3x3 = facenet_utils.conv2d_bn(inception_5a,
                                       layer='inception_5b_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = facenet_utils.conv2d_bn(inception_5b_pool,
                                        layer='inception_5b_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1))
    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = facenet_utils.conv2d_bn(inception_5a,
                                       layer='inception_5b_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    if include_norm:
        norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)
        return Model(inputs=[myInput], outputs=norm_layer)
    return Model(inputs=[myInput], outputs=dense_layer)


#######################################################################
# Function: scale_cnn
# Description: Takes an existing keras CNN model and scales the network
#              so that it will images of some multiple k larger than the
#              original model's input size. Does this by adding MaxPooling
#              layers to the end of the network preserving the original
#              network' output size
# Parameters:
#           - input_model: original model to be scaled
#           - k_val: value to scale the model by
#           -
#           -
#           -
#           -
#           -
# Returns: A new Keras Model object an input shape of k_value*()
# Notes:
#######################################################################
# def scale_cnn(input_model, k_val):
#
#     # Add k_value MaxPooling layers
#     for k in range(k_val):
#         # Add MaxPool layer to the sequential model
#     # TODO: Implement based on Catherine's code

def facenet_dense_layers(num_classes, input_shape=(128,)):
    model = Sequential()
    #model.add(Dense(128, activation='relu', input_shape=input_shape, name="dense_128"))
    model.add(Dense(num_classes, activation='softmax'))
    # input_layer = Input(input_shape)
    # x = Dense(128, activation='relu')(input_layer)
    # x = Dense(num_classes, activation='softmax')(x)
    return model


#def resnet_dense_layers():
    # TODO
    # This should be cool
