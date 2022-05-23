import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow_addons as tfa
from AdamW import AdamOptimizer
import GatingNN
from GatingNN import GatingNN as Gating
from ExpertWeights import ExpertWeights
from AdamParameter import AdamParameter

# AdamOptimizer
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer#migrate-to-tf2
# https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW

def Normalize(X, axis):
    Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)

    for i in range(Xstd.size):
        if (Xstd[i] == 0):
            Xstd[i] = 1

    return (X - Xmean) / Xstd


class MANN:
    def __init__(self, num_joints, num_styles,
                 rng, sess,
                 dataPath, num_experts,
                 gating_hidden_size, hidden_size,
                 gating_index,
                 batch_size, epoch, Te, Tmult,
                 init_learning_rate, init_weightDecay, init_keep_prob
                 ):
        self.num_joints = num_joints
        self.num_styles = num_styles

        self.rng = rng
        self.sess = sess
        self.num_experts = num_experts

        # load data
        self.input_data = Normalize(np.float32(np.loadtxt(dataPath + '/Input.txt')), axis=0)
        self.output_data = Normalize(np.float32(np.loadtxt(dataPath + '/Output.txt')), axis=0)
        self.input_size = self.input_data.shape[1]
        self.output_size = self.output_data.shape[1]
        self.size_data = self.input_data.shape[0]
        self.hidden_size = hidden_size

        # gating network
        self.gating_hidden_size = gating_hidden_size
        self.gating_index = gating_index

        # hyperparameter
        self.batch_size = batch_size
        self.epoch = epoch
        self.total_batch = int(self.size_data / self.batch_size)

        self.AP = AdamParameter(nEpochs=self.epoch,
                                 Te=Te, Tmult=Tmult,
                                 LR=init_learning_rate,
                                 weightDecay=init_weightDecay,
                                 batchSize=self.batch_size, nBatches=self.total_batch
                                 )

        self.init_keep_prob = init_keep_prob

    def build_model(self):
        self.nn_X = tf.placeholder(tf.float32, [self.batch_size, self.input_size],  name='nn_X')
        self.nn_Y = tf.placeholder(tf.float32, [self.batch_size, self.output_size], name='nn_Y')
        self.nn_keep_prob = tf.placeholder(tf.float32, name='nn_keep_prob')
        self.curr_learningRate = tf.placeholder(tf.float32, name='curr_learningRate')
        self.curr_weightDecay = tf.placeholder(tf.float32, name='curr_weightDecay')

        # Gating Network
        self.gating_input_size = len(self.gating_index)
        self.gating_input = tf.transpose(GatingNN.getInput(self.nn_X, self.gating_index))
        self.gatingNN = Gating(self.rng, self.gating_input, self.gating_input_size, self.num_experts, self.gating_hidden_size, self.nn_keep_prob)
        self.blending_coefficient = self.gatingNN.blending

        # experts
        self.layer0 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size,  self.input_size), name='layer0')
        self.layer1 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size, self.hidden_size), name='layer1')
        self.layer2 = ExpertWeights(self.rng, (self.num_experts, self.output_size, self.hidden_size), name='layer2')


        w0 = self.layer0.get_weight(self.blending_coefficient, self.batch_size)
        w1 = self.layer1.get_weight(self.blending_coefficient, self.batch_size)
        w2 = self.layer2.get_weight(self.blending_coefficient, self.batch_size)

        b0 = self.layer0.get_bias(self.blending_coefficient, self.batch_size)
        b1 = self.layer1.get_bias(self.blending_coefficient, self.batch_size)
        b2 = self.layer2.get_bias(self.blending_coefficient, self.batch_size)

        # Motion Prediction Network
        input_layer = tf.expand_dims(self.nn_X, -1)
        input_layer = tf.compat.v1.nn.dropout(input_layer, keep_prob=self.nn_keep_prob)

        hidden_layer0 = tf.matmul(w0, input_layer) + b0
        hidden_layer0 = tf.nn.elu(hidden_layer0)
        hidden_layer0 = tf.compat.v1.nn.dropout(hidden_layer0, keep_prob=self.nn_keep_prob)

        hidden_layer1 = tf.matmul(w1, hidden_layer0) + b1
        hidden_layer1 = tf.nn.elu(hidden_layer1)
        hidden_layer1 = tf.compat.v1.nn.dropout(hidden_layer1, keep_prob=self.nn_keep_prob)

        output_layer = tf.matmul(w2, hidden_layer1) + b2
        self.output_layer = tf.squeeze(output_layer, -1, name='output_layer')

        self.loss = tf.reduce_mean(tf.square(self.nn_Y - self.output_layer))
        # self.optimizer = tfa.optimizers.AdamW(
        #     weight_decay=self.curr_weightDecay,
        #     learning_rate=self.curr_learningWeight,
        #     epsilon=1e-8
        # ).minimize(self.loss, var_list=tf.GradientTape().watch_variables())
        self.optimizer = AdamOptimizer(learning_rate=self.curr_learningRate, wdc=self.curr_weightDecay).minimize(self.loss)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        """training"""
        print("total_batch:", self.total_batch)
        # randomly select training set
        I = np.arange(self.size_data)
        self.rng.shuffle(I)
        error_train = np.ones(self.epoch)

        print('Learning starts..')
        for epoch in range(self.epoch):
            avg_cost_train = 0
            for i in range(self.total_batch):
                index_train = I[i * self.batch_size:(i + 1) * self.batch_size]
                batch_xs = self.input_data[index_train]
                batch_ys = self.output_data[index_train]
                clr, wdc = self.AP.getParameter(epoch)  # currentLearningRate & weightDecayCurrent
                feed_dict = {self.nn_X: batch_xs, self.nn_Y: batch_ys, self.nn_keep_prob: self.init_keep_prob,
                             self.curr_learningRate: clr, self.curr_weightDecay: wdc}
                l, _, = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                avg_cost_train += l / self.total_batch

                if i % 1000 == 0:
                    print(f'batch: {i}/{self.total_batch}, training loss: {l}, clr: {clr}, wdc: {wdc}')

            # print and save training test error
            print('Epoch:', '%04d' % (epoch + 1), '/', self.epoch,  'trainingloss =', '{:.9f}'.format(avg_cost_train))

            error_train[epoch] = avg_cost_train

            # save model
            with open('./save/training_loss.txt', 'a') as f:
                f.write('Epoch: {:02d} / {:d} trainingloss = {:.9f}'.format(epoch+1, self.epoch, avg_cost_train))
                f.write('\n')

            if (epoch+1) % 5 == 0:
                save_file_name = "./save/model_ckpt_" + str(epoch + 1)
                saver.save(self.sess, save_file_name)
            else:
                saver.save(self.sess,  "./save/model_ckpt")
            

        print('Learning Finished')
