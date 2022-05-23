import numpy as np
import tensorflow as tf


class GatingNN:
    def __init__(self, rng, input_x,
                 input_size, output_size, hidden_size, nn_keep_prob):
        # random number generator
        self.rng = rng

        self.input = input_x

        # dropout 시, drop하지 않을 확률
        self.keep_prob = nn_keep_prob

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.w0 = tf.Variable(self.initial_weight([hidden_size, input_size]), name="gatingNN_w0")
        self.w1 = tf.Variable(self.initial_weight([hidden_size, hidden_size]), name="gatingNN_w1")
        self.w2 = tf.Variable(self.initial_weight([output_size, hidden_size]), name="gatingNN_w2")

        self.b0 = tf.Variable(tf.zeros([hidden_size, 1], tf.float32), name="gatingNN_b0")
        self.b1 = tf.Variable(tf.zeros([hidden_size, 1], tf.float32), name="gatingNN_b1")
        self.b2 = tf.Variable(tf.zeros([output_size, 1], tf.float32), name="gatingNN_b2")

        # blending coefficients
        self.blending = self.forwardPropagation()


    def initial_weight(self, shape):
        rng = self.rng
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        weight = np.asarray(rng.uniform(low=-weight_bound, high=weight_bound, size=shape), dtype=np.float32)

        return tf.convert_to_tensor(weight, dtype=tf.float32)

    # 참고 : https://kevinthegrey.tistory.com/131
    def forwardPropagation(self):
        input_layer = tf.compat.v1.nn.dropout(self.input, keep_prob=self.keep_prob)

        hidden_layer0 = tf.matmul(self.w0, input_layer) + self.b0
        hidden_layer0 = tf.nn.elu(hidden_layer0)
        hidden_layer0 = tf.compat.v1.nn.dropout(hidden_layer0, keep_prob=self.keep_prob)

        hidden_layer1 = tf.matmul(self.w1, hidden_layer0) + self.b1
        hidden_layer1 = tf.nn.elu(hidden_layer1)
        hidden_layer1 = tf.compat.v1.nn.dropout(hidden_layer1, keep_prob=self.keep_prob)

        output_layer = tf.matmul(self.w2, hidden_layer1) + self.b2
        output_layer = tf.compat.v1.nn.softmax(output_layer, dim=0)

        return output_layer

# gatingNN에서 사용하는 input은 x의 subset
# that are the foot end effector velocities, the current action variables and the desired velocity of the character
# get the velocity of joints, desired velocity and style
def getInput(data, index_joint):
    gating_input = data[..., index_joint[0]:index_joint[0]+1]

    index_joint.remove(index_joint[0])
    for i in index_joint:
        gating_input = tf.concat([gating_input, data[..., i:i+1]], axis=-1)

    return gating_input
