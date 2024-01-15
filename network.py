import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class QNetwork:
    def __init__(self, config, size, network_type='simple'):
        size = size
        channel = 3
        self.config = config
        self.n_class = len(self.config.new_class)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.variable_scope('input'):
            if network_type == 'simple':
                self.state = tf.compat.v1.placeholder(shape=[None, size], dtype=tf.float32)
            else:
                self.state = tf.compat.v1.placeholder(shape=[None, size, size, channel], dtype=tf.float32)
            self.learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
            self.target_q = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.reward = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.action = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32)
            self.terminal = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.target_soft_update = tf.compat.v1.placeholder(dtype=tf.float32)
        with tf.compat.v1.variable_scope('main_net'):
            if network_type == 'complex':
                self.q_mnet = self.build_complex_network()
            elif network_type == 'simple':
                self.q_mnet = self.build_simple_network()
            else:
                self.q_mnet = self.build_network()
        with tf.compat.v1.variable_scope('target_net'):
            if network_type == 'complex':
                self.q_tnet = self.build_complex_network()
            elif network_type == 'simple':
                self.q_tnet = self.build_simple_network()
            else:
                self.q_tnet = self.build_network()

        main_variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="main_net")
        target_variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        self.update_target = [t.assign((1 - self.target_soft_update) * t + self.target_soft_update * m)
                              for t, m in zip(target_variables, main_variables)]
        self.q_wrt_a = tf.expand_dims(tf.gather_nd(self.q_mnet, self.action, batch_dims=1), axis=1)
        self.target = self.reward + (1 - self.terminal) * self.config.gamma * self.target_q
        self.loss = tf.compat.v1.losses.huber_loss(self.target, self.q_wrt_a)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=main_variables)

    def build_network(self):
        x = tf.compat.v1.layers.conv2d(self.state, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d(x, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.layers.dense(x, 256, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q
    
    def build_simple_network(self):
        x = tf.compat.v1.layers.dense(self.state, 128, activation=tf.nn.relu)
        # x = tf.compat.v1.layers.dense(x, 64, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q

    def efficientnet_block(self, inputs, filters, kernel_size, strides=(1, 1), expand_ratio=1, se_ratio=None, activation=tf.nn.swish):
        # Depthwise Convolution
        x = tf.compat.v1.nn.depthwise_conv2d(inputs, filters, kernel_size, strides=strides, padding='same')
        x = tf.compat.v1.layers.BatchNormalization(x)
        x = activation(x)

        # Project layer
        input_filters = inputs.get_shape().as_list()[-1]
        expand_filters = input_filters * expand_ratio
        x = tf.compat.v1.layers.conv2d(x, expand_filters, (1, 1), padding='same', use_bias=False)
        x = tf.compat.v1.layers.batch_normalization(x)
        x = activation(x)

        # Squeeze and Excitation (SE) block
        if se_ratio:
            se_filters = max(1, int(input_filters * se_ratio))
            squeeze = tf.reduce_mean(x, [1, 2], keepdims=True)
            excitation = tf.compat.v1.layers.conv2d(squeeze, se_filters, (1, 1), activation=tf.nn.swish, use_bias=True)
            excitation = tf.sigmoid(excitation)
            x = tf.multiply(x, excitation)

        return x
    
    def build_complex_network(self):
        # Entry flow
        x = tf.compat.v1.layers.conv2d(self.state, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        
        # Middle flow
        for _ in range(3):
            x = tf.compat.v1.layers.conv2d(x, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        
        # Exit flow
        x = tf.compat.v1.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.compat.v1.layers.flatten(x)

        x = tf.compat.v1.layers.dense(x, 128, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q
