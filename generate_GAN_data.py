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



def decode_network():
    # Assuming that input shape is (1024,1024,3) and output shape is (512,)
    input_shape = (1024, 1024, 3)
    img_input = Input(input_shape)
    x = Conv2D(32, 3, padding='same')(img_input) # 1024,1024,3
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 512,512,32
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 256,256,32
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 128,128,64
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 64,64,64
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
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 4,4,128
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 2,2,128
    x = Flatten()(x)
    model = Model(inputs=img_input, outputs=x)
    return model




def run_decode_model(images, latents):
    # images, latents = data_in
    print(np.shape(images))
    print(np.shape(latents))
    decode_model = decode_network()
    decode_model.compile(Adam(), loss='mean_squared_error')
    decode_model.fit(images, latents, epochs=10, validation_split=0.1)



def save_img(data):
    img, path = data
    PIL.Image.fromarray(img, 'RGB').save(path)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()
    print(type(Gs).__name__)
    print(Gs.components)
    # Specify number of images generated
    total_num_imgs = 11000
    num_batch_imgs = 20
    num_save_imgs = 1000
    save_dir = os.path.join(os.getcwd(), 'GAN_images')
    os.makedirs(save_dir, exist_ok=True)
    images = None
    rnd = np.random.RandomState()
    # Create list of latent vectors

    latent_list = rnd.randn(total_num_imgs, Gs.input_shape[1])
    # Generate the paths to all the images
    png_filenames = []
    start_val = 9000
    for k in range(total_num_imgs):
        png_filenames.append(os.path.join(save_dir, str(start_val+k) + '.png'))

    # Save the values of the latent inputs
    if(os.path.exists(os.path.join(save_dir, 'latents_list.npy')) and start_val > 0):
        og_list = np.load(os.path.join(save_dir, 'latents_list.npy'))
        full_list = np.concatenate((og_list, latent_list), axis=0)
        np.save(os.path.join(save_dir, 'latents_list.npy'), full_list)
    else:
        np.save(os.path.join(save_dir, 'latents_list.npy'), latent_list)
    save_iter = 0
    print("Starting for loop")
    for i in range(int(total_num_imgs/num_batch_imgs)):

        latents = latent_list[i*num_batch_imgs:(i+1)*num_batch_imgs]
        # Generate image batch
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        if images is None:
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        else:
            images = np.append(images,
                           Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt),
                           axis=0)
        os.system('clear')
        print("Number of images created: " + str(len(images)))
        if(((i*num_batch_imgs) % num_save_imgs == 0) and i > 0) or i == (int(total_num_imgs/num_batch_imgs)-1):
            img_paths = png_filenames[save_iter*num_save_imgs:(save_iter + 1)*num_save_imgs]
            m_utils.multi_thread(save_img, zip(images, img_paths), os.cpu_count())
            images = None
            save_iter = save_iter + 1
            # Save the images and clear the image array


        # print(np.shape(images))
        # print("Starting save")
        # png_filenames = []
        # for k in range(num_batch_imgs):
        #     png_filenames.append(os.path.join(save_dir, str((i*num_batch_imgs) + k) + '.png'))
        # m_utils.multi_thread(save_img, zip(images, png_filenames), os.cpu_count())
        # print("Finished Save")
    # png_filenames = []
    # print("Starting Save Procedure")

    # m_utils.multi_thread(save_img, zip(images, png_filenames), os.cpu_count())
    # np.save(os.path.join(save_dir, 'latents_list.npy'), latent_list)


    # print("Results of imgs_eval")
    # print(np.shape(imgs_eval))
    # print(np.shape(imgs_eval[0]))
    #
    # imgs_eval = np.reshape(imgs_eval, (-1,1024,1024,3))
    # print(imgs_eval[0])
    # #imgs_eval = np.array(((imgs_eval + np.ones(np.shape(imgs_eval)))/2)*255,dtype=np.int)
    # print(imgs_eval[0])
    # png_filename = os.path.join(save_dir, 'example'+ str(99) +'.png')
    # PIL.Image.fromarray(imgs_eval[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
