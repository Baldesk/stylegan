# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Linear Separability (LS)."""

from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

classifier_urls = [
    'https://drive.google.com/uc?id=1Q5-AI6TwWhCVM7Muu4tBM7rp5nG_gmCX', # celebahq-target_model-00-male.pkl
    'https://drive.google.com/uc?id=1Q5c6HE__ReW2W8qYAXpao68V1ryuisGo', # celebahq-target_model-01-smiling.pkl
    'https://drive.google.com/uc?id=1Q7738mgWTljPOJQrZtSMLxzShEhrvVsU', # celebahq-target_model-02-attractive.pkl
    'https://drive.google.com/uc?id=1QBv2Mxe7ZLvOv1YBTLq-T4DS3HjmXV0o', # celebahq-target_model-03-wavy-hair.pkl
    'https://drive.google.com/uc?id=1QIvKTrkYpUrdA45nf7pspwAqXDwWOLhV', # celebahq-target_model-04-young.pkl
    'https://drive.google.com/uc?id=1QJPH5rW7MbIjFUdZT7vRYfyUjNYDl4_L', # celebahq-target_model-05-5-o-clock-shadow.pkl
    'https://drive.google.com/uc?id=1QPZXSYf6cptQnApWS_T83sqFMun3rULY', # celebahq-target_model-06-arched-eyebrows.pkl
    'https://drive.google.com/uc?id=1QPgoAZRqINXk_PFoQ6NwMmiJfxc5d2Pg', # celebahq-target_model-07-bags-under-eyes.pkl
    'https://drive.google.com/uc?id=1QQPQgxgI6wrMWNyxFyTLSgMVZmRr1oO7', # celebahq-target_model-08-bald.pkl
    'https://drive.google.com/uc?id=1QcSphAmV62UrCIqhMGgcIlZfoe8hfWaF', # celebahq-target_model-09-bangs.pkl
    'https://drive.google.com/uc?id=1QdWTVwljClTFrrrcZnPuPOR4mEuz7jGh', # celebahq-target_model-10-big-lips.pkl
    'https://drive.google.com/uc?id=1QgvEWEtr2mS4yj1b_Y3WKe6cLWL3LYmK', # celebahq-target_model-11-big-nose.pkl
    'https://drive.google.com/uc?id=1QidfMk9FOKgmUUIziTCeo8t-kTGwcT18', # celebahq-target_model-12-black-hair.pkl
    'https://drive.google.com/uc?id=1QthrJt-wY31GPtV8SbnZQZ0_UEdhasHO', # celebahq-target_model-13-blond-hair.pkl
    'https://drive.google.com/uc?id=1QvCAkXxdYT4sIwCzYDnCL9Nb5TDYUxGW', # celebahq-target_model-14-blurry.pkl
    'https://drive.google.com/uc?id=1QvLWuwSuWI9Ln8cpxSGHIciUsnmaw8L0', # celebahq-target_model-15-brown-hair.pkl
    'https://drive.google.com/uc?id=1QxW6THPI2fqDoiFEMaV6pWWHhKI_OoA7', # celebahq-target_model-16-bushy-eyebrows.pkl
    'https://drive.google.com/uc?id=1R71xKw8oTW2IHyqmRDChhTBkW9wq4N9v', # celebahq-target_model-17-chubby.pkl
    'https://drive.google.com/uc?id=1RDn_fiLfEGbTc7JjazRXuAxJpr-4Pl67', # celebahq-target_model-18-double-chin.pkl
    'https://drive.google.com/uc?id=1RGBuwXbaz5052bM4VFvaSJaqNvVM4_cI', # celebahq-target_model-19-eyeglasses.pkl
    'https://drive.google.com/uc?id=1RIxOiWxDpUwhB-9HzDkbkLegkd7euRU9', # celebahq-target_model-20-goatee.pkl
    'https://drive.google.com/uc?id=1RPaNiEnJODdr-fwXhUFdoSQLFFZC7rC-', # celebahq-target_model-21-gray-hair.pkl
    'https://drive.google.com/uc?id=1RQH8lPSwOI2K_9XQCZ2Ktz7xm46o80ep', # celebahq-target_model-22-heavy-makeup.pkl
    'https://drive.google.com/uc?id=1RXZM61xCzlwUZKq-X7QhxOg0D2telPow', # celebahq-target_model-23-high-cheekbones.pkl
    'https://drive.google.com/uc?id=1RgASVHW8EWMyOCiRb5fsUijFu-HfxONM', # celebahq-target_model-24-mouth-slightly-open.pkl
    'https://drive.google.com/uc?id=1RkC8JLqLosWMaRne3DARRgolhbtg_wnr', # celebahq-target_model-25-mustache.pkl
    'https://drive.google.com/uc?id=1RqtbtFT2EuwpGTqsTYJDyXdnDsFCPtLO', # celebahq-target_model-26-narrow-eyes.pkl
    'https://drive.google.com/uc?id=1Rs7hU-re8bBMeRHR-fKgMbjPh-RIbrsh', # celebahq-target_model-27-no-beard.pkl
    'https://drive.google.com/uc?id=1RynDJQWdGOAGffmkPVCrLJqy_fciPF9E', # celebahq-target_model-28-oval-face.pkl
    'https://drive.google.com/uc?id=1S0TZ_Hdv5cb06NDaCD8NqVfKy7MuXZsN', # celebahq-target_model-29-pale-skin.pkl
    'https://drive.google.com/uc?id=1S3JPhZH2B4gVZZYCWkxoRP11q09PjCkA', # celebahq-target_model-30-pointy-nose.pkl
    'https://drive.google.com/uc?id=1S3pQuUz-Jiywq_euhsfezWfGkfzLZ87W', # celebahq-target_model-31-receding-hairline.pkl
    'https://drive.google.com/uc?id=1S6nyIl_SEI3M4l748xEdTV2vymB_-lrY', # celebahq-target_model-32-rosy-cheeks.pkl
    'https://drive.google.com/uc?id=1S9P5WCi3GYIBPVYiPTWygrYIUSIKGxbU', # celebahq-target_model-33-sideburns.pkl
    'https://drive.google.com/uc?id=1SANviG-pp08n7AFpE9wrARzozPIlbfCH', # celebahq-target_model-34-straight-hair.pkl
    'https://drive.google.com/uc?id=1SArgyMl6_z7P7coAuArqUC2zbmckecEY', # celebahq-target_model-35-wearing-earrings.pkl
    'https://drive.google.com/uc?id=1SC5JjS5J-J4zXFO9Vk2ZU2DT82TZUza_', # celebahq-target_model-36-wearing-hat.pkl
    'https://drive.google.com/uc?id=1SDAQWz03HGiu0MSOKyn7gvrp3wdIGoj-', # celebahq-target_model-37-wearing-lipstick.pkl
    'https://drive.google.com/uc?id=1SEtrVK-TQUC0XeGkBE9y7L8VXfbchyKX', # celebahq-target_model-38-wearing-necklace.pkl
    'https://drive.google.com/uc?id=1SF_mJIdyGINXoV-I6IAxHB_k5dxiF6M-', # celebahq-target_model-39-wearing-necktie.pkl
]

#----------------------------------------------------------------------------

def prob_normalize(p):
    p = np.asarray(p).astype(np.float32)
    assert len(p.shape) == 2
    return p / np.sum(p)

def mutual_information(p):
    p = prob_normalize(p)
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    result = 0.0
    for x in range(p.shape[0]):
        p_x = px[x]
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            p_y = py[y]
            if p_xy > 0.0:
                result += p_xy * np.log2(p_xy / (p_x * p_y)) # get bits as output
    return result

def entropy(p):
    p = prob_normalize(p)
    result = 0.0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            if p_xy > 0.0:
                result -= p_xy * np.log2(p_xy)
    return result

def conditional_entropy(p):
    # H(Y|X) where X corresponds to axis 0, Y to axis 1
    # i.e., How many bits of additional information are needed to where we are on axis 1 if we know where we are on axis 0?
    p = prob_normalize(p)
    y = np.sum(p, axis=0, keepdims=True) # marginalize to calculate H(Y)
    return max(0.0, entropy(y) - mutual_information(p)) # can slip just below 0 due to FP inaccuracies, clean those up.

#----------------------------------------------------------------------------

class LS(metric_base.MetricBase):
    def __init__(self, num_samples, num_keep, attrib_indices, minibatch_per_gpu, **kwargs):
        assert num_keep <= num_samples
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_keep = num_keep
        self.attrib_indices = attrib_indices
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph for each GPU.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()

                # Generate images.
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                dlatents = Gs_clone.components.mapping.get_output_for(latents, None, is_validation=True)
                images = Gs_clone.components.synthesis.get_output_for(dlatents, is_validation=True, randomize_noise=True)

                # Downsample to 256x256. The attribute classifiers were built for 256x256.
                if images.shape[2] > 256:
                    factor = images.shape[2] // 256
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3, 5])

                # Run target_model for each attribute.
                result_dict = dict(latents=latents, dlatents=dlatents[:,-1])
                for attrib_idx in self.attrib_indices:
                    classifier = misc.load_pkl(classifier_urls[attrib_idx])
                    logits = classifier.get_output_for(images, None)
                    predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))
                    result_dict[attrib_idx] = predictions
                result_expr.append(result_dict)

        # Sampling loop.
        results = []
        for _ in range(0, self.num_samples, minibatch_size):
            results += tflib.run(result_expr)
        results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}

        # Calculate conditional entropy for each attribute.
        conditional_entropies = defaultdict(list)
        for attrib_idx in self.attrib_indices:
            # Prune the least confident samples.
            pruned_indices = list(range(self.num_samples))
            pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
            pruned_indices = pruned_indices[:self.num_keep]

            # Fit SVM to the remaining samples.
            svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
            for space in ['latents', 'dlatents']:
                svm_inputs = results[space][pruned_indices]
                try:
                    svm = sklearn.svm.LinearSVC()
                    svm.fit(svm_inputs, svm_targets)
                    svm.score(svm_inputs, svm_targets)
                    svm_outputs = svm.predict(svm_inputs)
                except:
                    svm_outputs = svm_targets # assume perfect prediction

                # Calculate conditional entropy.
                p = [[np.mean([case == (row, col) for case in zip(svm_outputs, svm_targets)]) for col in (0, 1)] for row in (0, 1)]
                conditional_entropies[space].append(conditional_entropy(p))

        # Calculate separability scores.
        scores = {key: 2**np.sum(values) for key, values in conditional_entropies.items()}
        self._report_result(scores['latents'], suffix='_z')
        self._report_result(scores['dlatents'], suffix='_w')

#----------------------------------------------------------------------------
