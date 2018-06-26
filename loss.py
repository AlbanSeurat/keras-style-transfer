from keras import backend as K


class LossFunction():

    def __init__(self, batch_size, lamda):
        self.batch_size = batch_size
        self.lamda = lamda

    @staticmethod
    def mse(coded, base):
        return K.mean(K.square(coded - base))

    @staticmethod
    def sse(coded, base):
        return K.sum(K.square(coded - base))

    def layer_style_loss(self, content, style):

        content_mean = K.mean(content, axis=[1, 2])
        content_var = K.sqrt(K.var(content, axis=[1, 2]) + 1e-03)

        style_mean = K.mean(style, axis=[1, 2])
        style_std = K.sqrt(K.var(style, axis=[1, 2]) + 1e-03)

        m_loss = LossFunction.sse(content_mean, style_mean) / self.batch_size
        s_loss = LossFunction.sse(content_var, style_std) / self.batch_size

        return m_loss + s_loss

    def style_loss(self, x):
        loss = K.variable(0., dtype='float32')
        for i in range(4):
            loss + self.layer_style_loss(x[i], x[i + 4])
        return loss

    def content_loss(self, x):
        return LossFunction.mse(x[0], x[1])

    def total_loss(self, x):
        return x[0] + self.lamda * x[1]
