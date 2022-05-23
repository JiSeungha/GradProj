import numpy as np
import tensorflow as tf


class ExpertWeights:
    def __init__(self, rng, shape, name):
        self.rng = rng

        self.weight_shape = shape
        self.bias_shape = (shape[0], shape[1], 1)

        self.alpha = tf.Variable(self.init_alpha(), name=name+'alpha')
        self.beta = tf.Variable(self.init_beta(), name=name+'beta')

    def init_alpha(self):
        shape = self.weight_shape
        rng = self.rng
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)

        return tf.convert_to_tensor(alpha, dtype=tf.float32)

    def init_beta(self):
        return tf.zeros(self.bias_shape, tf.float32)

    def get_weight(self, coefficient, batch_size):
        a = tf.expand_dims(self.alpha, 1)
        a = tf.tile(a, [1, batch_size, 1, 1])

        w = tf.expand_dims(coefficient, -1)
        w = tf.expand_dims(w, -1)

        return tf.reduce_sum(w*a, axis=0)

    def get_bias(self, coefficient, batch_size):
        b = tf.expand_dims(self.beta, 1)
        b = tf.tile(b, [1, batch_size, 1, 1])

        w = tf.expand_dims(coefficient, -1)
        w = tf.expand_dims(w, -1)

        return tf.reduce_sum(w*b, axis=0)




