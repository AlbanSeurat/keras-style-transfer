import keras.backend as K
from tensorflow.python import debug as tf_debug

from style import StyleTransfer


def main():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    style = StyleTransfer(stop_layer='block1_conv1')
    style.train("content", "style")


if __name__ == '__main__':
    main()