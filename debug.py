from IPython.display import SVG
from keras import backend as K 
import numpy.ma as ma
from keras.utils.vis_utils import model_to_dot
import pylab as pl
import numpy as np
from matplotlib import pyplot as plt


def _print_tensor(x, layer_name):
    return K.print_tensor(x, message=layer_name)

def dump_model(model):
    return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

def display_layer(layer):
    x1w = layer.get_weights()[0][:,:,:,0]
    plt.figure(figsize=(15, 15))
    for i in range(0,x1w.shape[2]):
        plt.subplot(8,8,i + 1)
        plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
    plt.show()