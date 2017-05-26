import numpy as np
from tf_network.tf_utils import *
from tf_network.expconfig.runtime_config import HeKaimingInitializer


# ---------- Activation ----------

class Activation(KeyMap):
	_map = {
		"relu": tf.nn.relu
	}


# ---------- Regularizer ----------

class Regularizer(KeyMap):
	_map = {
		"l2": lambda wdval: lambda x: wdval * tf.nn.l2_loss(x),
		"l2_mean": lambda wdval: lambda x: wdval * tf.reduce_mean(x * x)
	}


# ---------- Initializer ----------

class Initializer(KeyMap):
	_map = {
		"truncated_normal": tf.truncated_normal_initializer,
		"random_norm": tf.random_normal_initializer,
		"random_uniform": tf.random_uniform_initializer,
		"xavier": xavier_initializer,
		"variance_scaling": variance_scaling_initializer,
		"zero": tf.zeros_initializer,
		"one": tf.ones_initializer,
	}


# ---------- BatchNorm ----------

class BatchNormConfig:
	eps = 1e-5


# ---------- Layers ----------

class LinearLayer:
	activation = None
	stddev = None
	init_bias = None
	weight_decay = None
	init = None
	other_kwargs = None
	
	def __init__(self, activation=None, init=None, weight_decay=None, stddev=None, use_bn=False, init_bias=None,
				 keep_prob=None, **other_kwargs):
		self.activation = activation
		assert isinstance(init, dict), "'init' argument should be a dict, but type {} received.".format(type(init))
		self.init = init
		self.stddev = stddev
		self.init_bias = init_bias
		self.use_bn = use_bn
		if self.use_bn: other_kwargs.update({"use_bias": False})
		self.keep_prob = keep_prob
		if weight_decay:
			if isinstance(weight_decay, list):
				self.weight_decay = weight_decay
			else:
				assert isinstance(weight_decay,
								  float), "'weight_decay' should be of type float, but type {} received.".format(
					type(weight_decay))
				self.weight_decay = ["l2", weight_decay]
		self.other_kwargs = other_kwargs
	
	def __call__(self, h, index, training):
		if self.use_bn:
			h = batch_normalization(h, name=self.bn_name(index), **self.bn_kwargs)
		if self.activation is not None:
			h = Activation.get(self.activation)(h)
		if self.keep_prob is not None:
			h = tf.cond(training, lambda: tf.nn.dropout(h, self.keep_prob), lambda: h, name=self.dropout_name(index))
		return h
	
	@property
	def kernel_initializer(self):
		if self.init and "kernel" in self.init:
			return tf.constant_initializer(self.init["kernel"])
		elif isinstance(self.stddev, float):
			return tf.truncated_normal_initializer(stddev=self.stddev)
		elif isinstance(self.stddev, list):
			init_type, init_param_dict = self.stddev
			return Initializer.get(init_type)(**init_param_dict)
		elif self.stddev == "hekaiming":
			return HeKaimingInitializer()
		else:
			return xavier_initializer()
	
	@property
	def bias_initializer(self):
		if self.init and "bias" in self.init:
			return tf.constant_initializer(self.init["bias"])
		elif self.init_bias:
			return tf.constant_initializer(self.init_bias)
		else:
			return tf.zeros_initializer()
	
	@property
	def scale_initializer(self):
		if self.init and "scale" in self.init:
			return tf.constant_initializer(self.init["scale"])
		else:
			return tf.ones_initializer()
	
	@property
	def offset_initializer(self):
		if self.init and "offset" in self.init:
			return tf.constant_initializer(self.init["offset"])
		else:
			return tf.zeros_initializer()
	
	@property
	def kernel_regularizer(self):
		if self.weight_decay:
			wdtype, wdval = self.weight_decay
			return Regularizer.get(wdtype)(wdval)
	
	@property
	def common_linear_kwargs(self):
		return {
			# "activation": Activation.get(self.activation) if self.activation else None,
			"kernel_initializer": self.kernel_initializer,
			"use_bias": not self.use_bn,
			"bias_initializer": self.bias_initializer,
			"kernel_regularizer": self.kernel_regularizer,
			**self.other_kwargs
		}
	
	@property
	def bn_kwargs(self):
		return {
			"eps": BatchNormConfig.eps,
			"scale_initializer": self.scale_initializer,
			"offset_initializer": self.offset_initializer
		}
	
	@property
	def common_name_suffix(self):
		return "_{}".format(self.activation) if self.activation else ""
	
	def bn_name(self, index):
		return "BN_{}".format(self.name(index))
	
	def dropout_name(self, index):
		return "Dropout_kp{}_{}".format(self.keep_prob, self.name(index))
	
	@property
	def common_form_dict(self):
		return {
			"activation": self.activation,
			"use_bn": self.use_bn,
			"keep_prob": self.keep_prob,
			"stddev": self.stddev,
			"init_bias": self.init_bias,
			"weight_decay": self.weight_decay,
			**self.other_kwargs
		}
	
	def renew_init(self, sess, graph, scope, index):
		kernel_variable = graph.get_tensor_by_name("{}/{}/kernel:0".format(scope, self.name(index)))
		self.init = {"kernel": sess.run(kernel_variable)}
		if self.use_bn:
			bn_scale = graph.get_tensor_by_name("{}/{}/scale:0".format(scope, self.bn_name(index)))
			bn_offset = graph.get_tensor_by_name("{}/{}/offset:0".format(scope, self.bn_name(index)))
			self.init["bn_scale"] = sess.run(bn_scale)
			self.init["bn_offset"] = sess.run(bn_offset)
		else:
			bias_variable = graph.get_tensor_by_name("{}/{}/bias:0".format(scope, self.name(index)))
			self.init["bias"] = sess.run(bias_variable)
	
	# TODO Check
	def widen_bias(self, indices, magnifier):
		if self.init and "bias" in self.init:
			old_bias = self.init["bias"]
			self.init["bias"] = old_bias[indices] * magnifier
	
	# TODO Check
	@property
	def common_identity_kwargs(self):
		return {
			"activation": self.activation,
			"weight_decay": self.weight_decay,
			**self.other_kwargs
		}


class DenseLayer(LinearLayer):
	layer_name = "Dense"
	
	units = None
	
	def __init__(self, units, **kwargs):
		self.units = units
		super(self.__class__, self).__init__(**kwargs)
	
	@property
	def linear_kwargs(self):
		return {"units": self.units, **self.common_linear_kwargs}
	
	def name(self, index):
		return "Dense_i{}_u{}".format(index, self.units) + self.common_name_suffix
	
	def __call__(self, x, index, training=False):
		h = tf.layers.dense(x, name=self.name(index), **self.linear_kwargs)
		return super(self.__class__, self).__call__(h, index, training)
	
	@property
	def form_dict(self):
		return [
			self.__class__.layer_name,
			{"units": self.units, **self.common_form_dict}
		]
	
	# TODO Check
	def widen_kernel(self, indices, magnifier):
		if self.init and "kernel" in self.init:
			old_kernel = self.init["kernel"]
			self.init["kernel"] = old_kernel[:, indices] * magnifier
	
	# TODO Check
	def prev_widen_from_dense(self, indices):
		if self.init and "kernel" in self.init:
			old_kernel = self.init["kernel"]
			self.init["kernel"] = old_kernel[indices, :]
	
	# TODO Check
	def prev_widen_from_conv(self, prev_filters, indices):
		if self.init and "kernel" in self.init:
			old_kernel = self.init["kernel"].reshape((-1, prev_filters, self.units))
			self.init["kernel"] = old_kernel[:, indices, :].reshape((-1, self.units))
	
	# TODO Check
	def widen(self, cache):
		if cache.start == self:
			old_units = self.units
			self.units = Expander.get_next(self.__class__, old_units)
			indices, magnifier = Expander.random_expand(old_units, self.units)
			self.widen_kernel(indices, magnifier)
			self.widen_bias(indices, magnifier)
			return WidenCache(cache.start, indices)
		else:
			if isinstance(cache.start, DenseLayer):
				self.prev_widen_from_dense(cache.indices)
			else:
				self.prev_widen_from_conv(cache.prev_filters, cache.indices)
	
	# TODO Check
	def identity_transform(self):
		kernel = np.eye(self.units)
		return self.__class__(self.units, init={"kernel": kernel}, **self.common_identity_kwargs)


class ConvLayer(LinearLayer):
	layer_name = "Conv"
	
	filters = None
	kernel_size = None
	
	def __init__(self, filters, kernel_size, **kwargs):
		self.filters = filters
		self.kernel_size = kernel_size
		super(self.__class__, self).__init__(**kwargs)
	
	@property
	def linear_kwargs(self):
		return {"filters": self.filters, "kernel_size": self.kernel_size, **self.common_linear_kwargs}
	
	def name(self, index):
		kernel_size = 'x'.join([str(x) for x in self.kernel_size]) if isinstance(self.kernel_size,
																				 list) else self.kernel_size
		return "Conv_i{}_f{}_k{}".format(index, self.filters, kernel_size) + self.common_name_suffix
	
	def __call__(self, x, index, training=False):
		h = tf.layers.conv2d(x, name=self.name(index), **self.linear_kwargs)
		return super(self.__class__, self).__call__(h, index, training)
	
	@property
	def form_dict(self):
		return [
			self.__class__.layer_name,
			{"filters": self.filters, "kernel_size": self.kernel_size, **self.common_form_dict}
		]
	
	# TODO Check
	def widen_kernel(self, indices, magnifier):
		if self.init and "kernel" in self.init:
			old_kernel = self.init["kernel"]
			self.init["kernel"] = old_kernel[:, :, :, indices] * magnifier
	
	# TODO Check
	def prev_widen(self, indices):
		if self.init and "kernel" in self.init:
			old_kernel = self.init["kernel"]
			self.init["kernel"] = old_kernel[:, :, indices, :]
	
	# TODO Check
	def widen(self, cache):
		if cache.start == self:
			old_filters = self.filters
			self.filters = Expander.get_next(self.__class__, old_filters)
			indices, magnifier = Expander.random_expand(old_filters, self.filters)
			self.widen_kernel(indices, magnifier)
			self.widen_bias(indices, magnifier)
			return WidenCache(cache.start, indices, magnifier, old_filters)
		else:
			self.prev_widen(cache.indices)
	
	# TODO Check
	def identity_transform(self):
		mid = (self.kernel_size + 1) // 2
		kernel = np.zeros([self.kernel_size, self.kernel_size, self.filters, self.filters])
		kernel[mid, mid] = np.eye(self.filters)
		return self.__class__(self.filters, self.kernel_size, init={"kernel": kernel}, **self.common_identity_kwargs)


class PoolLayer:
	layer_name = "Pool"
	
	pool_size = None
	strides = None
	padding = None
	other_kwargs = None
	
	def __init__(self, pool_size, strides, padding="same", init=None, keep_prob=None, **other_kwargs):
		self.pool_size = pool_size
		self.strides = strides
		self.padding = padding
		self.keep_prob = keep_prob
		self.other_kwargs = other_kwargs
	
	@property
	def kwargs(self):
		return {
			"pool_size": self.pool_size,
			"strides": self.strides,
			"padding": self.padding,
			**self.other_kwargs
		}
	
	def name(self, index):
		return "Pool_i{}_p{}_s{}".format(index, self.pool_size, self.strides)
	
	def __call__(self, x, index, training=None):
		h = tf.layers.max_pooling2d(x, name=self.name(index), **self.kwargs)
		if self.keep_prob:
			dropout_name = "Dropout_{}_{}".format(self.keep_prob, self.name(index))
			h = tf.cond(training, lambda: tf.nn.dropout(h, self.keep_prob), lambda: tf.nn.dropout(h, 1),
						name=dropout_name)
		return h
	
	@property
	def form_dict(self):
		return [
			self.__class__.layer_name,
			{
				"keep_prob": self.keep_prob,
				**self.kwargs,
			},
		]
	
	@property
	def init(self):
		pass
	
	def renew_init(self, sess, graph, scope, index):
		pass
	
	def widen(self, cache):
		return cache


class FlattenLayer:
	layer_name = "Flatten"
	
	def __init__(self, init=None, keep_prob=None):
		self.keep_prob = keep_prob
	
	def name(self, index):
		return "Flatten_i{}".format(index)
	
	def __call__(self, x, index, training=None):
		h = tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])], name=self.name(index))
		if self.keep_prob:
			dropout_name = "Dropout_{}_{}".format(self.keep_prob, self.name(index))
			h = tf.cond(training, lambda: tf.nn.dropout(h, self.keep_prob), lambda: tf.nn.dropout(h, 1),
						name=dropout_name)
		return h
	
	@property
	def form_dict(self):
		return [
			self.__class__.layer_name,
			{"keep_prob": self.keep_prob}
		]
	
	@property
	def init(self):
		pass
	
	def renew_init(self, sess, graph, scope, index):
		pass
	
	def widen(self, cache):
		return cache


# class BatchNormalizationLayer:
# 	layer_name = "BatchNorm"
#
# 	init = None
#
# 	def __init__(self, init=None, **other_kwargs):
# 		if init:
# 			assert isinstance(init, dict), "'init' argument should be a dict, but type {} received.".format(type(init))
# 			self.init = init
# 		self.other_kwargs = other_kwargs
#
# 	"""
# 	moving_mean & moving_variance
# 	These two variable's constant initializer is removed
# 		because I don't think their value should be transfered
# 			between different training & testing loops,
# 		even if using net2net transformation.
# 	"""
#
# 	@property
# 	def beta_initializer(self):
# 		if self.init and "beta" in self.init:
# 			return tf.constant_initializer(self.init["beta"])
# 		else:
# 			return tf.zeros_initializer()
#
# 	@property
# 	def gamma_initializer(self):
# 		if self.init and "gamma" in self.init:
# 			return tf.constant_initializer(self.init["gamma"])
# 		else:
# 			return tf.ones_initializer()
#
# 	@property
# 	def kwargs(self):
# 		return {
# 			"beta_initializer": self.beta_initializer,
# 			"gamma_initializer": self.gamma_initializer,
# 			**self.other_kwargs
# 		}
#
# 	def name(self, index):
# 		return "BatchNorm_i{}".format(index)
#
# 	def __call__(self, x, index, training=None):
# 		return tf.layers.batch_normalization(x, name=self.name(index), training=training, **self.kwargs)
#
# 	@property
# 	def form_dict(self):
# 		return [
# 			self.__class__.layer_name,
# 			self.other_kwargs
# 		]
#
# 	def renew_init(self, graph, scope, index):
# 		beta_variable = graph.get_tensor_by_name("{}/{}/beta:0".format(scope, self.name(index)))
# 		gamma_variable = graph.get_tensor_by_name("{}/{}/gamma:0".format(scope, self.name(index))),
# 		self.init = {
# 			"beta": sess.run(beta_variable),
# 			"gamma": sess.run(gamma_variable)
# 		}
#
# 	def widen_beta(self, indices, magnifier):
# 		if self.init and "beta" in self.init:
# 			old_beta = self.init["beta"]
# 			self.init["beta"] = old_beta[indices] * magnifier
#
# 	def widen_gamma(self, indices):
# 		if self.init and "gamma" in self.init:
# 			old_gamma = self.init["gamma"]
# 			self.init["gamma"] = old_gamma[indices]
#
# 	def widen(self, cache):
# 		indices, magnifier = cache.indices, cache.magnifier
# 		self.widen_beta(cache.indices, cache.magnifier)
# 		self.widen_gamma(cache.indices)
# 		return cache

# class DropoutLayer:
# 	layer_name = "Dropout"
# 	keep_prob = None
#
# 	def __init__(self, keep_prob, init=None):
# 		self.keep_prob = keep_prob
#
# 	def name(self, index):
# 		return "Dropout_i{}".format(index) + "_kp_{}".format(self.keep_prob)
#
# 	def __call__(self, x, index, training=True):
# 		return tf.cond(training, lambda: tf.nn.dropout(x, self.keep_prob), lambda: tf.nn.dropout(x, 1), name=self.name(index))
#
# 	@property
# 	def form_dict(self):
# 		return [
# 			self.__class__.layer_name,
# 			{"keep_prob": self.keep_prob}
# 		]
#
# 	@property
# 	def init(self):
# 		pass
#
# 	def renew_init(self, sess, graph, scope, index):
# 		pass
#
# 	def widen(self, cache):
# 		return cache

_lst = [
	DenseLayer,
	ConvLayer,
	PoolLayer,
	FlattenLayer,
]


class Layer(KeyMap):
	_map = {x.layer_name: x for x in _lst}


# ---------- Net2Net ----------

class WidenCache:
	start = None
	indices = None
	magnifier = None
	prev_filters = None
	
	def __init__(self, start, indices=None, magnifier=None, prev_filters=None):
		self.start = start
		self.indices = indices
		self.magnifier = magnifier
		self.prev_filters = prev_filters


class Expander:
	@staticmethod
	def make_magnifier(units, indices):
		l = np.zeros(units)
		for x in indices:
			l[x] += 1
		return (1.0 / l)[indices]
	
	@staticmethod
	def random_expand(units, new_units):
		base_indices = np.arange(units)
		indices = np.concatenate([base_indices, np.random.choice(base_indices, new_units - units)])
		return indices, Expander.make_magnifier(units, indices)
	
	grid = {
		ConvLayer: [32, 64, 96, 128, 192, 256],
		DenseLayer: [128, 256, 512, 1024]
	}
	
	@staticmethod
	def get_next(layer_type, current_value):
		for x in Expander.grid[layer_type]:
			if x > current_value:
				return x
		return current_value
