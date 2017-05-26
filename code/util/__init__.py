__all__ = ["create_session", "global_step"]

import tensorflow as tf

def create_session(graph):
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth=True
	return tf.Session(graph=graph, config=config)

def global_step():
	return tf.get_variable(name="global_step", dtype=tf.int32, shape=(), trainable=False, initializer=tf.zeros_initializer(), collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
