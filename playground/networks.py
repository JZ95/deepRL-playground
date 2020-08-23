import tensorflow.compat.v1 as tf

def conv_net(conv_in, dim_out, name='conv_net', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        for _ in range(3):
            conv_out = tf.layers.conv2d(conv_in, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
            pool_out = tf.layers.max_pooling2d(conv_out, pool_size=3, strides=2)
            conv_in = pool_out

        flatten = tf.layers.flatten(pool_out)
        x = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        return tf.layers.dense(x, dim_out)


def ffn(x, hidden_dims, activations, name='ffn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        for i, (dim, act_fn) in enumerate(zip(hidden_dims, activations), start=1):
            x = tf.layers.dense(x, dim, activation=act_fn, name="layer_%d" % i)
    return x
