from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import ZeroPadding2D, BatchNormalization
from debug import _print_tensor
import tensorflow as tf

class AdaIN(Layer):

    def __init__(self, epsilon=1e-5, **kwargs):
        self.epsilon = epsilon
        self.alpha = 0.5
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaIN, self).build(input_shape)
        
    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('AdaIN must be called on a list of tensors '
                            '(exactly 2). Got: ' + str(inputs))
        content_layer = inputs[0]
        style_layer = inputs[1]

        style_mean, style_variance = tf.nn.moments(style_layer, [1, 2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_layer, [1, 2], keep_dims=True)

        normalized_content_features = tf.nn.batch_normalization(content_layer, content_mean,
                                                                content_variance, style_mean,
                                                                tf.sqrt(style_variance), self.epsilon)
        self.result = self.alpha * normalized_content_features + (1 - self.alpha) * content_layer
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    

# Extending the ZeroPadding2D layer to do reflection padding instead.
class ReflectionPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        (top_pad, bottom_pad), (left_pad, right_pad) = self.padding
        pattern = [[0, 0],
                   [top_pad, bottom_pad],
                   [left_pad, right_pad],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')


