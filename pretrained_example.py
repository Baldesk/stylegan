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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow as tf
import tensorboard
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import cv2
from training import misc


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
    #num_imgs = 10

    #Gs.convert('G', num_inputs=1, )
    # input_tnsr = tf.get_default_graph().get_tensor_by_name(Gs.input_names[0])
    # # input_tnsr = tf.get_variable(Gs.input_names[0])
    # print(input_tnsr)
    #print(Gs.list_ops())

    # # Pick latent vector.
    # #rnd = np.random.RandomState(5)
    # rnd = np.random.RandomState()
    # latents = rnd.randn(num_imgs, Gs.input_shape[1])
    # print(Gs.input_shape[1])
    # print("Gs.num_inputs" + str(Gs.num_inputs))
    # print(Gs.input_names)
    # latent_input = tf.Variable(tf.random_normal(shape=[Gs.input_shape[1]]), dtype=tf.float32)
    # #l_input = [latent_input, None]
    # #print(len(latent_input))
    # output_stylenetwork = Gs.get_output_for(latent_input, None)
    # siamese_adv_input = Input(shape=(Gs.output_shape), tensor=output_stylenetwork)
    #Gs_clone = Gs.clone()
    #latents = tf.Variable(tf.random_normal([1] + Gs_clone.input_shape[1:]), name="learned_latent")
    latents = tf.Variable(tf.random_normal([1] + Gs.input_shape[1:]), name="learned_latent")
    #latents = tf.Variable(tf.random_normal([Gs.input_shape[1]]))


    # dlatents = Gs_clone.components.mapping.get_output_for(latents, None, is_validation=True)
    # images = Gs_clone.components.synthesis.get_output_for(dlatents, is_validation=True, randomize_noise=True)
    dlatents = Gs.components.mapping.get_output_for(latents, None, is_validation=True)
    images = Gs.components.synthesis.get_output_for(dlatents, is_validation=True, randomize_noise=True)
    scale_factor = tf.constant(255/2)
    shift_factor = tf.constant(0.5 - -1 * (255/2), dtype=tf.float32)

    images_transposed = tf.transpose(images, [0, 2, 3, 1])
    images_out = tf.add(tf.multiply(tf.cast(images_transposed, tf.float32), scale_factor), shift_factor)
    images_out_int = tf.saturate_cast(images_out, tf.uint8)
    print(images_out.shape)

    #images_out = tflib.convert_images_to_uint8(images, nchw_to_nhwc=True)
    #images_out = tf.to_int32(images)
    #print(tf.get_gradien)
    # latent_layer = Input(shape=Gs_clone.input_shape[1:], tensor=latents)
    # output_imgs = Lambda(images)
    # style_model = Model(inputs=[latent_layer], outputs=[output_imgs])
    # style_model.summary()
    #input("BROOOO HOLD UP")

    #Initialize latent variables
    init_latent = tf.initializers.variables([latents])

    # Define Comparison Operation
    compare_img_in = tf.placeholder(tf.float32, shape=(None,1024,1024,3), name='compare_img')
    #out_image_resize = tf.image.resize_bilinear(images_out, (600,600), name='img_resize')
    dist = tf.sqrt(tf.to_float(tf.reduce_sum(tf.squared_difference(compare_img_in, images_out))), name='dist_op')


    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(dist, var_list=(latents))
    init_opt = tf.initializers.variables(optimizer.variables())


    tf.keras.backend.get_session().run(init_latent) # Initialize latent variables
    tf.keras.backend.get_session().run(init_opt) # Initialize training variables
    #tf.keras.backend.get_session().run(tf.initializers.variables(optimizer.variables()))

    train_iter = 1000
    cmp_img = np.array(cv2.cvtColor(cv2.imread('/home/kyle/Downloads/kyle_passport_square.png'),
                                    cv2.COLOR_BGR2RGB),
                                    dtype=np.float32)
    cmp_img = cv2.resize(cmp_img, (1024,1024))
    png_filename = os.path.join(config.result_dir, 'cmp_img' + '.png')
    PIL.Image.fromarray(np.array(cmp_img, dtype=np.uint8), 'RGB').save(png_filename)
    feed_image_dict = {
        compare_img_in: np.reshape(cmp_img, (1,1024,1024,3))
    }

    # For the number of training iterations
    for i in range(train_iter):
        # Setup Images


        distance_val = tf.keras.backend.get_session().run(dist, feed_dict=feed_image_dict)
        print("Iteration " + str(i+1) + " distance: " + str(distance_val))
        # Train on batch
        tf.keras.backend.get_session().run(train_step, feed_dict=feed_image_dict)






    imgs_eval = tf.keras.backend.get_session().run(images_out_int)

    print("Results of imgs_eval")
    print(np.shape(imgs_eval))
    print(np.shape(imgs_eval[0]))

    imgs_eval = np.reshape(imgs_eval, (-1,1024,1024,3))
    print(imgs_eval[0])
    #imgs_eval = np.array(((imgs_eval + np.ones(np.shape(imgs_eval)))/2)*255,dtype=np.int)
    print(imgs_eval[0])
    png_filename = os.path.join(config.result_dir, 'example'+ str(99) +'.png')
    PIL.Image.fromarray(imgs_eval[0], 'RGB').save(png_filename)


    #out_arrays = [np.empty([num_items] + tfutil.shape_to_list(expr.shape)[1:], expr.dtype.name) for expr in out_expr]


    # print(images)
    # #gen_in = Input(shape=(Gs_clone.input_shape), tensor=latents)
    # #styleGan_model = Model(inputs=[latents], outputs=images)
    # print(Gs_clone.output_shape[1:])
    # # This is output of StyleGAN
    # after_styleGAN_in = Input(Gs_clone.output_shape[1:], tensor=images) # [None,3,1024,1024]
    # #x = Dense(5)(after_styleGAN_in)
    # model = Model(inputs=[after_styleGAN_in], outputs=[x])
    # model.summary()
    #
    # t_board = TensorBoard(log_dir='logs_styleGAN')
    # model.compile(optimizer=Adam(), loss="mean_squared_error")
    # x = 42
    # model.predict(42, steps=1)



    #model.fit()
    #styleGan_model.summary()

    # Generate image.
    #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    #images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    # os.makedirs(config.result_dir, exist_ok=True)
    #
    # for k in range(num_imgs):
    #     png_filename = os.path.join(config.result_dir, 'example'+ str(k) +'.png')
    #     PIL.Image.fromarray(images[k], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
