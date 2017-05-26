import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer, variance_scaling_initializer


class KeyMap:
	_map = {}
	
	@classmethod
	def get(cls, key):
		assert key in cls._map, "{} {} not found.".format(cls.__name__, key)
		return cls._map[key]


def rank(x):
	return len(x.shape.as_list())


def batch_normalization(x, name=None, eps=1e-5, scale_initializer=None, offset_initializer=None):
	name = name or "BatchNorm"
	with tf.variable_scope(name):
		channel = x.shape[-1].value
		scale_initializer = scale_initializer or tf.ones_initializer()
		offset_initializer = offset_initializer or tf.zeros_initializer()
		
		scale = tf.get_variable(name="scale", initializer=scale_initializer, shape=[channel])
		offset = tf.get_variable(name="offset", initializer=offset_initializer, shape=[channel])
		mean, variance = tf.nn.moments(x, list(range(rank(x) - 1)), keep_dims=False)
		
		return tf.nn.batch_normalization(x, mean, variance, offset=offset, scale=scale, variance_epsilon=eps)


def dropout(x, name=None, training=False, keep_prob=None):
	name = name or "Dropout"
	return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob), lambda: x, name=name)


# get optimizer
def build_optimizer(optimizer_config):
	if optimizer_config["type"] == "adam":
		return tf.train.AdamOptimizer(optimizer_config["learning_rate"])
	elif optimizer_config["type"] == "rmsprop":
		return tf.train.RMSPropOptimizer(optimizer_config["learning_rate"])
	else:
		return tf.train.GradientDescentOptimizer(optimizer_config["learning_rate"])


# activation function
def activation_func(func_name, inputs):
	if func_name == "sigmoid":
		return tf.nn.sigmoid(inputs)
	elif func_name == "tanh":
		return tf.nn.tanh(inputs)
	elif func_name == "relu":
		return tf.nn.relu(inputs)
	elif func_name == "softmax":
		return tf.nn.softmax(inputs)
	else:
		return inputs


# build initializer
# variance_scaling_initializer:
# * To get [Delving Deep into Rectifiers](
#      http://arxiv.org/pdf/1502.01852v1.pdf), use (Default):<br/>
#     `factor=2.0 mode='FAN_IN' uniform=False`
#   * To get [Convolutional Architecture for Fast Feature Embedding](
#      http://arxiv.org/abs/1408.5093), use:<br/>
#     `factor=1.0 mode='FAN_IN' uniform=True`
#   * To get [Understanding the difficulty of training deep feedforward neural
#     networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
#     use:<br/>
#     `factor=1.0 mode='FAN_AVG' uniform=True.`
#   * To get `xavier_initializer` use either:<br/>
#     `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
#     `factor=1.0 mode='FAN_AVG' uniform=False`.
def build_initializer(initializer_config):
	if initializer_config["type"] == "truncated_normal":
		return tf.truncated_normal_initializer(stddev=initializer_config["stddev"])
	elif initializer_config["type"] == "random_normal":
		return tf.random_normal_initializer(stddev=initializer_config["stddev"])
	elif initializer_config["type"] == "random_uniform":
		return tf.random_uniform_initializer(-initializer_config["stddev"], initializer_config["stddev"])
	elif initializer_config["type"] == "xavier":
		return xavier_initializer(uniform=initializer_config["uniform"])
	elif initializer_config["type"] == "variance_scaling":
		return variance_scaling_initializer(initializer_config["factor"], initializer_config["mode"],
											initializer_config["uniform"])
	elif initializer_config["type"] == "zero":
		return tf.zeros_initializer()
	elif initializer_config["type"] == "one":
		return tf.ones_initializer()


# function: build embedding layer
# allow reuse
def embedding_layer(inputs, embedding_config, initializer=None, reuse=False, scope="Embedding"):
	with tf.variable_scope(scope, reuse=reuse):
		vocab_size, embedding_dim = embedding_config["vocab_size"], embedding_config["embedding_dim"]
		embedding = tf.get_variable("embedding", [vocab_size, embedding_dim], tf.float32, initializer)
		n_input = embedding_dim
		embedding_output = tf.nn.embedding_lookup(embedding, inputs)
	return embedding_output, n_input


# function: build recurrent cell, (not create variables)
def build_cell(cell_type, hidden_units, initializer=None, num_layers=1):
	if num_layers > 1:
		cell = rnn.MultiRNNCell([build_cell(cell_type, hidden_units, initializer, 1) for _ in range(num_layers)])
	else:
		if cell_type == "tf_lstm":
			cell = rnn.LSTMCell(hidden_units, initializer=initializer)
		elif cell_type == "tf_gru":
			cell = rnn.GRUCell(hidden_units)
		else:
			cell = rnn.BasicRNNCell(hidden_units)
	return cell


# function: build rnn encoder
# allow reuse
def rnn_encoder(encoder_inputs, seq_len, encoder_config, cell_initializer, reuse=False, scope="Encoder"):
	# Prepare data shape to match rnn function requirements
	# Current data input shape: (seq_num, num_steps, n_input)
	# Required shape: 'num_steps' tensors list of shape (seq_num, n_input)
	num_steps, n_input = encoder_inputs.shape.as_list()[1:3]
	encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
	encoder_inputs = tf.reshape(encoder_inputs, [-1, n_input])
	encoder_inputs = tf.split(encoder_inputs, num_steps, 0)
	
	with tf.variable_scope(scope, reuse=reuse):
		cell_type, hidden_units = encoder_config["cell_type"], encoder_config["hidden_units"]
		layer_num = encoder_config["num_layers"]
		if encoder_config["rnn_type"] == "BiRNN":
			# encoder_outputs' shape: 'num_steps' tensors list of shape (seq_num, fw || bw)
			fw_cell = build_cell(cell_type, hidden_units, cell_initializer, layer_num)
			bw_cell = build_cell(cell_type, hidden_units, cell_initializer, layer_num)
			cell = (fw_cell, bw_cell)
			encoder_outputs, encoder_state_fw, encoder_state_bw = \
				rnn.static_bidirectional_rnn(fw_cell, bw_cell, encoder_inputs, dtype=tf.float32,
											 sequence_length=seq_len, scope=encoder_config["rnn_type"])
			encoder_states = (encoder_state_fw, encoder_state_bw)
		else:
			# encoder_outputs' shape: 'num_steps' tensors list of shape (seq_num, hidden_units)
			cell = build_cell(cell_type, hidden_units, cell_initializer, layer_num)
			encoder_outputs, encoder_states = rnn.static_rnn(
				cell, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, scope=encoder_config["rnn_type"])
	return encoder_outputs, encoder_states, cell


# function: multi-dense layers with one output layer (without activation)
# allow reuse
def classifier(inputs, n_out, dense_configs, dense_initializer, output_initializer, reuse=False, out_bias=True, scope="classifier"):
	with tf.variable_scope(scope, reuse=reuse):
		inputs = dense_layers(inputs, dense_configs, dense_initializer)
		outputs = tf.layers.dense(inputs, n_out, kernel_initializer=output_initializer["kernel"],
								  bias_initializer=output_initializer["bias"], use_bias=out_bias, name="Output")
	return outputs


# function: multi-dense layers
# dense_config: {"units": ..., "batch_norm": ..., "activation_func": ...}
def dense_layers(inputs, configs, dense_initializer):
	for i, dense_config in enumerate(configs, start=1):
		with tf.variable_scope("Dense_{}".format(i)):
			if dense_config.get("batch_norm"):
				use_bias = False
			else:
				use_bias = True
			inputs = tf.layers.dense(inputs, dense_config["units"], kernel_initializer=dense_initializer["kernel"],
									 bias_initializer=dense_initializer["bias"], use_bias=use_bias)
			if dense_config.get("batch_norm"):
				inputs = batch_normalization(inputs, name="BN", eps=1e-5)
			if dense_config.get("activation_func"):
				inputs = activation_func(dense_config["activation_func"], inputs)
			else:
				inputs = activation_func("relu", inputs)
	return inputs


# function: return a variable for global step
def get_global_step(return_update_op=False):
	global_step = tf.get_variable(name="global_step", dtype=tf.int32, shape=(), trainable=False,
								  initializer=tf.zeros_initializer(),
								  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
	if return_update_op:
		global_step_update_op = tf.assign_add(global_step, tf.constant(1, dtype=tf.int32), "global_step_update_op")
		return global_step, global_step_update_op
	else:
		return global_step


# function: build the training loop, monitor loss, metric during training process
def train_loop(opt_op, task, initializers, inputs, overheads, update_ops, summaries):
	with tf.variable_scope("train") as scope:
		avg_loss, update_avg_loss = tf.metrics.mean(overheads["loss"], name="avg_loss")
		if "regression" in task:
			rmse, update_rmse = tf.metrics.root_mean_squared_error(inputs["labels"], overheads["predictions"])
			update_ops["train"] = [opt_op, update_avg_loss, update_rmse]
			summaries["train"] = tf.summary.merge([tf.summary.scalar("avg_loss_summary", avg_loss),
												   tf.summary.scalar("rmse_summary", rmse)])
		elif "classification" in task:
			accuracy, update_accuracy = tf.metrics.accuracy(inputs["labels"], overheads["predictions"])
			update_ops["train"] = [opt_op, update_avg_loss, update_accuracy]
			summaries["train"] = tf.summary.merge([tf.summary.scalar("avg_loss_summary", avg_loss),
												   tf.summary.scalar("accuracy_summary", accuracy)])
		initializers["train"] = [x.initializer for x in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope.name)]


# function: build the test (validation) loop, monitor loss, metric during test (validate) process
def test_loop(task, initializers, inputs, overheads, update_ops, summaries, stage="test"):
	with tf.variable_scope(stage) as scope:
		if "regression" in task:
			rmse, update_rmse = tf.metrics.root_mean_squared_error(inputs["labels"], overheads["predictions"])
			update_ops[stage] = [update_rmse, ]
			summaries[stage] = tf.summary.merge([tf.summary.scalar("rmse_summary", rmse), ])
			overheads["{}_metric".format(stage)] = rmse
		elif "classification" in task:
			accuracy, update_accuracy = tf.metrics.accuracy(inputs["labels"], overheads["predictions"])
			update_ops[stage] = [update_accuracy, ]
			summaries[stage] = tf.summary.merge([tf.summary.scalar("accuracy_summary", accuracy), ])
			overheads["{}_metric".format(stage)] = accuracy
		initializers[stage] = [var.initializer for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope.name)]


# function: return a mask matrix
# lower_bound <= index < upper_bound: True, 1.0
# else: False, 0.0
def matrix_mask(last_dim, lower_bound, upper_bound, dtype=tf.float32):
	upper_mask = tf.sequence_mask(upper_bound, last_dim)
	lower_mask = tf.logical_not(tf.sequence_mask(lower_bound, last_dim))
	mix_mask = tf.logical_and(upper_mask, lower_mask)
	mix_mask = tf.cast(mix_mask, dtype=dtype)
	return mix_mask
