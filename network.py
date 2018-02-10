import os
import json
import tensorflow as tf



tf.logging.set_verbosity(tf.logging.INFO)


NETWORK_HYPERPARAMS = {}


class Network(object):



    def __init__(self, input_size, learning_rate=0.001):

        rows, cols, depth = input_size

        # self.input_layer = tf.placeholder(tf.uint8,
        self.input_layer = tf.placeholder(tf.float32,
            shape=[None, rows, cols, depth],
            name='Network_Input')

        # First hidden layer
        conv_1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=16,
            kernel_size=[8, 8],
            strides=(4, 4),
            padding='SAME',
            activation=tf.nn.relu
        )

        # Second hidden layer
        conv_2 = tf.layers.conv2d(
            inputs=conv_1,
            filters=32,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding='SAME',
            activation=tf.nn.relu
        )

        conv_2_flat = tf.contrib.layers.flatten(conv_2)

        # Third hidden layer
        dense_3 = tf.layers.dense(inputs=conv_2_flat, units=256, activation=tf.nn.relu)

        # Output layer
        self.output_layer = tf.layers.dense(inputs=dense_3, units=2, activation=None)

        # Do I need to specify shape?
        self.target = tf.placeholder(tf.float32, name='Network_Target')

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.output_layer, self.target)))

        self.optm = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(init)


    def evaluate(self, input_data):
        return self.sess.run(self.output_layer, {self.input_layer: input_data})


    def train(self, input_data, target):
        return self.sess.run(self.optm, {self.input_layer: input_data, self.target: target})


    def error(self, input_data, target):
        return self.sess.run(self.loss, {self.input_layer: input_data, self.target: target})


    def save(self, path):

        save_path_model = self.saver.save(self.sess, os.path.join(path, 'model.ckpt'))
        print('Model saved in file: {}'.format(save_path_model))

        # save_path_hyperparameters = os.path.join(path, 'hyperparams.json')
        # with open(save_path_hyperparameters, 'w') as outfile:
        #     outfile.write(json.dumps(self.params, indent=4))
        # print('Model Hyperparameters saved in file: {}'.format(save_path_hyperparameters))

        return path

