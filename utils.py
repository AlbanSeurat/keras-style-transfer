from os import scandir
from os.path import join

import tensorflow as tf
import numpy as np
from keras import layers
from keras.applications import vgg19
from keras.callbacks import Callback
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize

from layers import ReflectionPadding2D


class NBatchLogger(Callback):
    def __init__(self, display=100):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print('\n{0} - Batch Loss: {1}'.format(self.seen, logs["loss"]))


def mirror_model(source_model, x):
    for pos in range(len(source_model.layers) - 1, 0, -1):
        layer = source_model.layers[pos]
        # x = Lambda(lambda y : _print_tensor(y, layer.name), name = layer.name + "_debug")(x)
        if type(layer) == Conv2D:
            x = ReflectionPadding2D(padding=((1, 1), (1, 1)), name=layer.name + "_padding")(x)
            act_func = None if pos == 1 else 'relu'
            x = Conv2D(layer.input_shape[3], layer.kernel_size, activation=act_func, \
                       padding='valid', name="decoder_" + layer.name)(x)
        elif type(layer) == MaxPooling2D:
            x = UpSampling2D(size=layer.pool_size, name=layer.name.replace("pool", "upsampling"))(x)
    return x


def clone_model(src_model, trainable=False, prefix="", output_names=None):
    outputs = []
    for layer in src_model.layers:
        if type(layer) == layers.InputLayer:
            x = input_layer = Input(batch_shape=layer.input_shape, name=prefix + layer.name)
        else:
            new_layer = layers.deserialize({'class_name': layer.__class__.__name__, 'config': layer.get_config()})
            new_layer.name = prefix + layer.name
            new_layer.trainable = trainable
            x = new_layer(x)
            if output_names is not None and layer.name in output_names:
                outputs.append(x)
            new_layer.set_weights(layer.get_weights())

    return input_layer, x if output_names is None else outputs


def preload_img(path, shape=None):
    img = load_img(path)
    if shape is not None:
        img = imresize(img, shape)
    img_array = img_to_array(img)
    return vgg19.preprocess_input(img_array)

def image_postprocess(x):
    x = x.copy()
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def list_images(path, limit=10, shape=None):
    i = 0
    for content_entry in scandir(path):
        img = preload_img(join(path, content_entry.name), shape)
        if i >= limit:
            return
        i += 1
        yield img

def list_batch_images(path_content, path_style, batch_size, limit=10, shape=None):

    contents = list_images(path_content, limit, shape)
    styles = list_images(path_style, limit, shape)

    while True:
        try:
            content_imgs = []
            style_imgs = []
            for h in range(batch_size):
                content_imgs.append(next(contents))
                style_imgs.append(next(styles))
            content_nparray_imgs = np.asarray(content_imgs)
            sytle_nparray_imgs = np.asarray(style_imgs)

            yield ([content_nparray_imgs, sytle_nparray_imgs], content_nparray_imgs)
        except StopIteration:
            break