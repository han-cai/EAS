import numpy as np, tensorflow as tf
from tf_network.tf_utils import KeyMap, batch_normalization, dropout


class RuntimeBaseLayer:
	_keep_prob = None
	_init = None
	
	def __init__(self, std_layer):
		self._keep_prob = std_layer.keep_prob
		self._init = std_layer.init
	
	def __call__(self, x, index, training):
		with tf.variable_scope(self.name(index)):
			x = self.body(x, index, training)
			x = self.dropout_call(x, index, training)
		return x
	
	# name
	
	def name(self, index):
		return self.__class__.layer_type + "_" + str(index)
	
	# dropout_call
	
	def dropout_call(self, x, index, training):
		return dropout(x, training=training, keep_prob=self._keep_prob) if self._keep_prob is not None else x


# ---------- RuntimeLinearLayer ----------

from tensorflow.python.ops.init_ops import Initializer
from tensorflow.contrib.layers import xavier_initializer, variance_scaling_initializer


class HeKaimingInitializer(Initializer):
	def __call__(self, shape, dtype=None, partition_info=None):
		return tf.random_normal(shape, stddev=np.sqrt(2.0 / np.prod(shape[:-1])))


class Initializer(KeyMap):
	_map = {
		"truncated_normal": tf.truncated_normal_initializer,
		"random_norm": tf.random_normal_initializer,
		"random_uniform": tf.random_uniform_initializer,
		"xavier": xavier_initializer,
		"variance_scaling": variance_scaling_initializer,
		"zero": tf.zeros_initializer,
		"one": tf.ones_initializer,
		"hekaiming": HeKaimingInitializer
	}


class Regularizer(KeyMap):
	_map = {
		None: lambda x: None,
		"l2": lambda wdval: lambda x: wdval * tf.nn.l2_loss(x),
		"l2_mean": lambda wdval: lambda x: wdval * tf.reduce_mean(x * x)
	}


class Activation(KeyMap):
	_map = {
		None: lambda x: x,
		"relu": tf.nn.relu
	}


class RuntimeLinearLayer(RuntimeBaseLayer):
	_stddev = None
	_weight_decay = None
	_activation = None
	_use_bn = None
	_init_bias = None
	
	def __init__(self, std_layer):
		super(RuntimeLinearLayer, self).__init__(std_layer)
		self._stddev = std_layer.stddev
		self._weight_decay = std_layer.weight_decay
		self._activation = std_layer.activation
		self._use_bn = std_layer.use_bn
		self._init_bias = std_layer.init_bias
	
	def body(self, x, index, training):
		x = self.linear_call(x, index, training)
		x = self.bn_call(x, index, training)
		x = self.activation(x)
		return x
	
	# linear_call
	
	def linear_call(self, x, index, training):
		pass
	
	@property
	def linear_kwargs(self):
		return {
			"kernel_initializer": self.kernel_initializer,
			"use_bias": self.use_bias,
			"bias_initializer": self.bias_initializer,
			"kernel_regularizer": self.kernel_regularizer
		}
	
	@property
	def kernel_initializer(self):
		init = self._init.get("kernel")
		if init is not None:
			return tf.constant_initializer(init)
		else:
			init_type, init_param = self._stddev
			return Initializer.get(init_type)(**init_param)
	
	@property
	def use_bias(self):
		return not self.use_bn
	
	@property
	def bias_initializer(self):
		init = self._init.get("bias")
		if init is not None:
			return tf.constant_initializer(init)
		else:
			return tf.constant_initializer(self._init_bias)
	
	@property
	def kernel_regularizer(self):
		reg_type, weight_decay = self._weight_decay
		return Regularizer.get(reg_type)(weight_decay)
	
	@property
	def linear_name(self):
		pass
	
	# bn_call
	
	def bn_call(self, x, index, training):
		return batch_normalization(x, name=self.bn_name, **self.bn_kwargs) if self.use_bn else x
	
	@property
	def use_bn(self):
		return self._use_bn
	
	@property
	def bn_kwargs(self):
		return {
			"scale_initializer": self.scale_initializer,
			"offset_initializer": self.offset_initializer
		}
	
	@property
	def scale_initializer(self):
		init = self._init.get("scale")
		if init is not None:
			return tf.constant_initializer(init)
		else:
			return tf.ones_initializer()
	
	@property
	def offset_initializer(self):
		init = self._init.get("offset")
		if init is not None:
			return tf.constant_initializer(init)
		else:
			return tf.zeros_initializer()
	
	@property
	def bn_name(self):
		return "BatchNorm"
	
	# activation
	
	@property
	def activation(self):
		return Activation.get(self._activation)


class RuntimeConvLayer(RuntimeLinearLayer):
	layer_type = "Conv"
	
	_filters = None
	_kernel_size = None
	_padding = None
	
	def __init__(self, std_layer):
		super(RuntimeConvLayer, self).__init__(std_layer)
		self._filters = std_layer.filters
		self._kernel_size = std_layer.kernel_size
		self._padding = std_layer.padding
	
	# linear_call
	
	def linear_call(self, x, index, training):
		return tf.layers.conv2d(x, name=self.linear_name, **self.linear_kwargs)
	
	@property
	def linear_kwargs(self):
		return {
			"filters": self._filters,
			"kernel_size": self._kernel_size,
			"padding": self._padding,
			**super(RuntimeConvLayer, self).linear_kwargs
		}
	
	@property
	def linear_name(self):
		return "Conv_f{}_k{}".format(self._filters, self._kernel_size)


class RuntimeDenseLayer(RuntimeLinearLayer):
	layer_type = "Dense"
	
	_units = None
	
	def __init__(self, std_layer):
		super(RuntimeDenseLayer, self).__init__(std_layer)
		self._units = std_layer.units
	
	# linear_call
	
	def linear_call(self, x, index, training):
		return tf.layers.dense(x, name=self.linear_name, **self.linear_kwargs)
	
	@property
	def linear_kwargs(self):
		return {
			"units": self._units,
			**super(RuntimeDenseLayer, self).linear_kwargs
		}
	
	@property
	def linear_name(self):
		return "Dense_u{}".format(self._units)


class RuntimePoolLayer(RuntimeBaseLayer):
	layer_type = "Pool"
	
	_pool_size = None
	_strides = None
	_padding = None
	
	def __init__(self, std_layer):
		super(RuntimePoolLayer, self).__init__(std_layer)
		self._pool_size = std_layer.pool_size
		self._strides = std_layer.strides
		self._padding = std_layer.padding
	
	def body(self, x, index, training):
		return tf.layers.max_pooling2d(x, name=self.pool_name, **self.pool_kwargs)
	
	@property
	def pool_kwargs(self):
		return {
			"pool_size": self._pool_size,
			"strides": self._strides,
			"padding": self._padding,
		}
	
	@property
	def pool_name(self):
		pool_size = "{}x{}".format(*self._pool_size) if isinstance(self._pool_size, list) else self._pool_size
		strides = "{}x{}".format(*self._strides) if isinstance(self._strides, list) else self._strides
		return "Pool_p{}_s{}".format(pool_size, strides)


class RuntimeFlattenLayer(RuntimeBaseLayer):
	layer_type = "Flatten"
	
	def body(self, x, index, training):
		return tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])], name="Flatten")


class RuntimeLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
		RuntimeConvLayer,
		RuntimeDenseLayer,
		RuntimePoolLayer,
		RuntimeFlattenLayer
	]
	}


class RuntimeLayersConfig(list):
	def __init__(self, std_layers_config):
		for std_layer in std_layers_config:
			self.append(RuntimeLayer.get(std_layer.__class__.layer_type)(std_layer))
	
	def __call__(self, x, training):
		for i, layer in enumerate(self):
			x = layer(x, i, training)
		return x


# ---------- RuntimeDividedConfig ----------

class RuntimeDividedConfig:
	runtime_monitor_config = None
	runtime_data_config = None
	runtime_network_config = None
	
	def __init__(self, std_divided_config):
		self.runtime_monitor_config = RuntimeMonitorConfig(std_divided_config.monitor_config)
		self.runtime_data_config = RuntimeDataConfig(std_divided_config.data_config)
		self.runtime_network_config = RuntimeNetworkConfig(std_divided_config.network_config)


# ---------- MonitorConfig ----------

class RuntimeMonitorConfig:
	training_loop = None
	validation_loop = None
	
	def __init__(self, monitor_config):
		self.training_loop = monitor_config.training_loop
		self.validation_loop = monitor_config.validation_loop


# ---------- DataConfig ----------

from util.data_processing import BasicReader


class Scheme(KeyMap):
	_map = {
		"TF": BasicReader.SCHEME_TF,
		"DENSENET": BasicReader.SCHEME_DENSENET,
		"OTHER": BasicReader.SCHEME_OTHER,
		"NONE": BasicReader.SCHEME_NONE
	}


class RuntimeDataConfig:
	batch_size = None
	train_full = None
	epochs = None
	scheme = None
	
	def __init__(self, data_config):
		self.batch_size = data_config.batch_size
		self.train_full = data_config.train_full
		self.epochs = data_config.epochs
		self.scheme = Scheme.get(data_config.scheme)


# ---------- NetworkConfig ----------

class Optimizer(KeyMap):
	_map = {
		"adam": tf.train.AdamOptimizer,
		"sgd": tf.train.GradientDescentOptimizer,
		"momentum": tf.train.MomentumOptimizer,
	}


class DecayType(KeyMap):
	_map = {
		"exp": lambda learning_rate, global_step, decay_param: tf.train.exponential_decay(learning_rate, global_step, **decay_param),
		"piecewise": lambda learning_rate, global_step, decay_param: tf.train.piecewise_constant(global_step, **decay_param)
	}


class RegType(KeyMap):
	_map = {
		"sum": tf.add_n,
		"mean": lambda reg_losses: tf.reduce_mean(tf.stack(reg_losses))
	}


class RuntimeNetworkConfig:
	_reg_type = None
	_image_size = None
	_layers = None
	_minimize = None
	
	def __init__(self, network_config):
		self._reg_type = network_config.reg_type
		self._image_size = network_config.image_size
		self._layers = RuntimeLayersConfig(network_config.layers)
		self._minimize = network_config.minimize
	
	@property
	def reg_type(self):
		return RegType.get(self._reg_type)
	
	@property
	def image_size(self):
		image_size = self._image_size
		return [image_size, image_size]
	
	@property
	def layers(self):
		return self._layers
	
	def apply(self, x, training):
		return self.layers(x, training)
	
	@property
	def minimize(self):
		optimizer, learning_rate, kwargs, decay = self._minimize
		optimizer = Optimizer.get(optimizer)
		
		def minimize_op(loss, global_step):
			lr = learning_rate
			if decay:
				decay_type, decay_param = decay
				lr = DecayType.get(decay_type)(learning_rate, global_step, decay_param)
			return optimizer(learning_rate=lr, **kwargs).minimize(loss, global_step)
		
		return minimize_op
