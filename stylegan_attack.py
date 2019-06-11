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
import cv2
import multi_process_utils as m_utils



def main():
    save_dir = os.path.join(os.getcwd(), 'GAN_images')
    latent_list = np.load(os.path.join(save_dir, 'latents_list.npy'))
    total_num_imgs = 1000
    paths = []
    for k in range(total_num_imgs):
        paths.append(os.path.join(save_dir, str(k) + '.png'))

    images = m_utils.multi_process(m_utils.get_img, paths, os.cpu_count())

    images = np.array(images)
    print(np.shape(images))
    print(np.shape(latent_list))
    decode_model = decode_network()
    decode_model.compile(Adam(), loss='mean_squared_error')
    decode_model.fit(images, latent_list, epochs=10, validation_split=0.1, batch_size=4)
    decode_model.save('decode_model.h5')


if __name__ == "__main__":
    main()
