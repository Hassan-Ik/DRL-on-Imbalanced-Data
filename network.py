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
        x = tf.compat.v1.layers.dense(self.state, 256, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q

    def efficientnetb0_block(self, inputs, filters, kernel_size, strides, expansion_factor):
        # Expansion phase
        x = tf.compat.v1.layers.conv2d(inputs, filters, (1, 1), padding='same', activation=tf.nn.relu)
        x = tf.compat.v1.layers.batch_normalization(x)
        

        # Depthwise convolution
        x = tf.compat.v1.nn.depthwise_conv2d(x, kernel_size, strides=strides, padding='same')
        x = tf.compat.v1.layers.batch_normalization(x)
        
        # Pointwise convolution
        x = tf.compat.v1.layers.conv2d(x, filters, (1, 1), padding='same', activation=tf.nn.relu)
        x = tf.compat.v1.layers.batch_normalization(x)

        # Skip connection if the shape is the same (identity)
        # if tf.shape(inputs)[-1] == filters:
        #     x = tf.keras.layers.Add()([inputs, x])
        return x    
    def build_complex_network(self):
        x = tf.compat.v1.layers.conv2d(self.state, 32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.compat.v1.layers.batch_normalization(x)
        
        # x = self.efficientnetb0_block(x, 16, (1, 1), (1, 1), 1)
        # x = self.efficientnetb0_block(x, 24, (3, 3), (2, 2), 6)
        # x = self.efficientnetb0_block(x, 40, (3, 3), (2, 2), 6)
        # x = self.efficientnetb0_block(x, 80, (3, 3), (2, 2), 6)
        x = tf.compat.v1.layers.conv2d(self.state, 16, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d(x, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.compat.v1.layers.average_pooling2d(x, (2,2), strides=2)

        x = tf.compat.v1.layers.conv2d(x, 64, 5, strides=1, activation=tf.nn.relu)
        x = tf.compat.v1.layers.conv2d(x, 128, 5, strides=1, activation=tf.nn.relu)
        x = tf.compat.v1.layers.average_pooling2d(x, (2,2), strides=2)
        
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.layers.dense(x, 256, activation=tf.nn.relu)
        x = tf.compat.v1.layers.dense(x, 128, activation=tf.nn.relu)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q
