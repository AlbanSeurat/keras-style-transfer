from keras.applications import vgg19
from keras.layers import Input, Lambda
from keras.models import Model

from loss import LossFunction
from utils import mirror_model, clone_model


class Vgg19TruncatedModel(Model):

    def __init__(self, stop_layer):
        _vgg19model = vgg19.VGG19(include_top=False)
        vgg19_output = _vgg19model.get_layer(stop_layer).output

        super(Vgg19TruncatedModel, self).__init__(inputs=_vgg19model.input, outputs=vgg19_output, name="vgg19")


class EncoderModel(Model):

    def __init__(self, src_model, trainable=False):
        content_input, content_output = clone_model(src_model, trainable, prefix="content_")
        style_input, styte_output = clone_model(src_model, trainable, prefix="style_")

        super(EncoderModel, self).__init__(inputs=[content_input, style_input],
                                           outputs=[content_output, styte_output], name="encoder_model")


class DecoderModel(Model):

    def __init__(self, src_model, input_shape):
        input_layer = Input(batch_shape=input_shape, name="decoder_input")
        decoder_layer = mirror_model(src_model, input_layer)
        super(DecoderModel, self).__init__(inputs=[input_layer], outputs=[decoder_layer], name="decoder_model")


class ReEncoderModel(Model):

    def __init__(self, src_model):

        relu_layers = ['block1_conv1',
                       'block2_conv1',
                       'block3_conv1',
                       'block4_conv1']

        content_input, content_outputs = clone_model(src_model, prefix="reencode_content_", output_names=relu_layers)
        style_input, style_outputs = clone_model(src_model, prefix="reencode_style_", output_names=relu_layers)

        super(ReEncoderModel, self).__init__(inputs=[content_input, style_input],
                                             outputs=content_outputs +  style_outputs, name="re_encoder_model")


class LossModel(Model):

    def __init__(self, re_encoder_model, adain_shape, batch_size, lamda=1e-2):

        self.loss = LossFunction(batch_size, lamda)
        adain_input = Input(batch_shape=adain_shape, name="adain_input")

        content_output = re_encoder_model.outputs[3]

        content_loss_layer = Lambda(self.loss.content_loss, name="content_loss")([content_output, adain_input])
        style_loss_layer = Lambda(self.loss.style_loss, name="style_loss")(re_encoder_model.outputs)
        total_loss_layer = Lambda(self.loss.total_loss, name="total_loss")([content_loss_layer, style_loss_layer])

        super(LossModel, self).__init__(inputs=re_encoder_model.inputs + [adain_input],
                                        outputs=[total_loss_layer])
