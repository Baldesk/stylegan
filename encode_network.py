# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Conv2D, MaxPooling2D, LeakyReLU, Flatten
import tensorflow as tf
import tensorboard
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import cv2
import multi_process_utils as m_utils
from tensorflow.keras.utils import multi_gpu_model
import math
from training import misc
from tensorflow.keras.applications.inception_v3 import InceptionV3

KYLE_IMG_PATH = '/home/kyle/Downloads/kyle_passport_square.png'
JONAH_HILL_PATH = '/home/kyle/Downloads/GAN_imgs/jonah_hill_ex.jpg'



def inception_encode_network():
    # LOAD STYLEGAN
    # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
    # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    #     _G, _D, Gs = pickle.load(f)


    #LOAD INCEPTION NETWORK
    # inception_v3_features.pkl
    #inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn')

    #inception_in = Input(shape=)
    model = InceptionV3(include_top=False, weights=None, input_shape=(256,256,3))
    x = Flatten()(model.output)
    x = Dense(512, activation='sigmoid')(x)
    new_model = Model(inputs=model.input, outputs=x)
    new_model.summary()
    return new_model






def decode_network(input_shape=(1024, 1024, 3)):
    # Assuming that input shape is (1024,1024,3) and output shape is (512,)
    img_input = Input(input_shape)
    x = Conv2D(16, 3, padding='same')(img_input) # 1024,1024,3
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 512,512,32
    x = Conv2D(16, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 256,256,32
    x = Conv2D(16, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 128,128,64
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 64,64,64
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 32,32,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 16,16,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 8,8,64
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 4,4,128
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 2,2,128
    x = Flatten()(x)
    model = Model(inputs=img_input, outputs=x)
    return model


def encode_network_256(input_shape=(256, 256, 3)):
    # Assuming that input shape is (1024,1024,3) and output shape is (512,)
    img_input = Input(input_shape)
    x = Conv2D(32, 3, padding='same')(img_input) # 256,256,3
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, 3, padding='same')(img_input) # 256,256,32
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 128,128,32
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 64,64,32
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 32,32,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 16,16,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 8,8,64
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 4,4,64
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 2,2,128
    x = Flatten()(x)
    model = Model(inputs=img_input, outputs=x)
    return model



def decode_network_128(input_shape=(128, 128, 3)):
    # Assuming that input shape is (128,128,3) and output shape is (512,)
    img_input = Input(input_shape)
    x = Conv2D(32, 3, padding='same')(img_input) # 256,256,3
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 128,128,32
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 64,64,32
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 32,32,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 16,16,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 8,8,64
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 4,4,64
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 2,2,128
    x = Flatten()(x)
    model = Model(inputs=img_input, outputs=x)
    return model







class styleSequence(Sequence):
    def __init__(self, img_paths, latents, batch_size):
        self.paths = img_paths
        self.y = latents
        self.batch_size = batch_size
    def __len__(self):
        return math.ceil(len(self.paths)/self.batch_size)

    def __getitem__(self, idx):
        path_list = self.paths[idx*self.batch_size:(idx+1)*self.batch_size]
        img_batch = np.array(m_utils.multi_process(m_utils.get_img_256, path_list, os.cpu_count()), dtype=np.float16)
        return (img_batch, self.y[idx*self.batch_size:(idx+1)*self.batch_size])



class styleSequence_preloaded(Sequence):
    def __init__(self, imgs, latents, batch_size):
        self.x = imgs
        self.y = latents
        self.batch_size = batch_size
    def __len__(self):
        return math.ceil(len(self.x)/self.batch_size)

    def __getitem__(self, idx):
        img_batch = np.array(self.x[idx*self.batch_size:(idx+1)*self.batch_size], dtype=np.float32)
        return (img_batch, self.y[idx*self.batch_size:(idx+1)*self.batch_size])



def main():
    save_dir = os.path.join(os.getcwd(), 'GAN_images_256')
    latent_list = np.load(os.path.join(save_dir, 'latents_list.npy'))
    total_num_imgs = 19980
    paths = []
    for k in range(total_num_imgs):
        paths.append(os.path.join(save_dir, str(k) + '.png'))

    images = m_utils.multi_process(m_utils.get_img_256, paths, os.cpu_count())


    #encode_model = encode_network_256()
    encode_model = inception_encode_network()


    multi_decode_model = multi_gpu_model(encode_model, 2)
    #multi_decode_model = encode_model
    multi_decode_model.compile(Adam(), loss='mean_absolute_error')

    #train_seq = styleSequence(paths, latent_list[:total_num_imgs], batch_size=32)
    train_seq = styleSequence_preloaded(images, latent_list[:total_num_imgs], batch_size=32)
    multi_decode_model.fit_generator(train_seq, epochs=5)
    encode_model.save('encode_model_256.h5')

    encode_model = load_model('encode_model_256.h5')
    # Load in image of Myself to get predicted output latent


    cmp_img = np.array(cv2.cvtColor(cv2.imread(JONAH_HILL_PATH),
                                    cv2.COLOR_BGR2RGB),
                                    dtype=np.float32)
    cmp_img = np.expand_dims(cv2.resize(cmp_img, (256,256)), axis=0)
    #cmp_img = np.expand_dims(cv2.resize(cmp_img, (1024,1024)), axis=0)

    latent_prediction = encode_model.predict(cmp_img)
    #np.save(os.path.join(os.getcwd(), 'latent_prediction_256.npy'), latent_prediction)
    np.save(os.path.join(os.getcwd(), 'latent_prediction_1024.npy'), latent_prediction)


if __name__ == "__main__":
    main()
