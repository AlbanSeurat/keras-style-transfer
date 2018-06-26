import torchfile
from keras.layers import Conv2D
from keras.models import Model

from layers import AdaIN
from models import EncoderModel, DecoderModel, Vgg19TruncatedModel, LossModel, ReEncoderModel
from utils import list_batch_images


class StyleTransfer():

    def __init__(self, stop_layer='block4_conv1'):
        self.src_model = Vgg19TruncatedModel(stop_layer)

        self.encoder_model = EncoderModel(self.src_model)

        self.adain_layer = AdaIN()(self.encoder_model.outputs)
        self.decoder_model = DecoderModel(self.src_model, self.adain_layer.shape.as_list())
        self.decoder_output = self.decoder_model(self.adain_layer)
        self.predict_model = Model(inputs=self.encoder_model.inputs, outputs=self.decoder_output)


    def load_weights(self, t7file):
        t7 = torchfile.load(t7file, force_8bytes_long=True)

        conv2Dlayers = [layer for layer in self.decoder_model.layers if type(layer) == Conv2D]
        pos = 0
        for idx, module in enumerate(t7.modules):

            if module._typename == b'nn.SpatialConvolution':
                weight = module.weight.transpose([2, 3, 1, 0])
                bias = module.bias
                #strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'
                conv2Dlayers[pos].set_weights([weight, bias])
                pos += 1

                #net = tf.nn.conv2d(net, weight, strides, padding='VALID')
                #net = tf.nn.bias_add(net, bias)
                #layers.append(net)

    def training(self, batch_size=8, lamda=1e-02):
        self.batch_size = batch_size

        style_input = self.encoder_model.inputs[1]

        self.re_encoder_model = ReEncoderModel(self.src_model)

        self.loss_model = LossModel(self.re_encoder_model, self.adain_layer.shape.as_list(), self.batch_size, lamda)

        loss_output = self.loss_model([self.decoder_output, style_input, self.adain_layer])
        self.train_model = Model(inputs=self.encoder_model.inputs, outputs=[loss_output])

        self.train_model.compile(optimizer='adam', loss=lambda x, loss: loss)

        return self

    def start(self, content_dir, style_dir, input_size=16, limit=128, shape=(32, 32, 3), epochs=10):

        self.train_model.fit_generator(list_batch_images(content_dir, style_dir, self.batch_size, limit=limit, shape=shape),
                                       epochs=epochs, verbose=1, steps_per_epoch=input_size / self.batch_size, shuffle=True)

    def predict(self, contents, styles):
        return self.predict_model.predict([contents, styles])
