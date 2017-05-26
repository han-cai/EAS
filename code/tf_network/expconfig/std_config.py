import json
import pickle as pkl
from tf_network.expconfig.basic_config import BasicConfig
from tf_network.layers import KeyMap


class StdBaseLayer:
	keep_prob = None
	init = None
	
	def __init__(self, basic_layer):
		self.keep_prob = basic_layer.keep_prob
		self.init = basic_layer.init
	
	def get_config(self):
		return [self.__class__.layer_type, self.get_layer_config()]
	
	def get_layer_config(self):
		return {
			"keep_prob": self.keep_prob
		}
	
	def get_init(self):
		return self.init


class StdLinearLayer(StdBaseLayer):
	stddev = None
	weight_decay = None
	activation = None
	use_bn = None
	init_bias = None
	
	def __init__(self, basic_layer):
		super(StdLinearLayer, self).__init__(basic_layer)
		self.stddev = basic_layer.stddev
		self.weight_decay = basic_layer.weight_decay
		self.activation = basic_layer.activation
		self.use_bn = basic_layer.use_bn
		self.init_bias = basic_layer.init_bias
	
	def get_layer_config(self):
		return {
			**super(StdLinearLayer, self).get_layer_config(),
			"stddev": self.stddev,
			"weight_decay": self.weight_decay,
			"activation": self.activation,
			"use_bn": self.use_bn,
			"init_bias": self.init_bias
		}


class StdConvLayer(StdLinearLayer):
	layer_type = "Conv"
	
	filters = None
	kernel_size = None
	padding = None
	
	def __init__(self, basic_layer):
		super(StdConvLayer, self).__init__(basic_layer)
		self.filters = basic_layer.filters
		self.kernel_size = basic_layer.kernel_size
		self.padding = basic_layer.padding
	
	def get_layer_config(self):
		return {
			**super(StdConvLayer, self).get_layer_config(),
			"filters": self.filters,
			"kernel_size": self.kernel_size,
			"padding": self.padding
		}


class StdDenseLayer(StdLinearLayer):
	layer_type = "Dense"
	
	units = None
	
	def __init__(self, basic_layer):
		super(StdDenseLayer, self).__init__(basic_layer)
		self.units = basic_layer.units
	
	def get_layer_config(self):
		return {
			**super(StdDenseLayer, self).get_layer_config(),
			"units": self.units
		}


class StdPoolLayer(StdBaseLayer):
	layer_type = "Pool"
	
	pool_size = None
	strides = None
	padding = None
	
	def __init__(self, basic_layer):
		super(StdPoolLayer, self).__init__(basic_layer)
		self.pool_size = basic_layer.pool_size
		self.strides = basic_layer.strides
		self.padding = basic_layer.padding
	
	def get_layer_config(self):
		return {
			**super(StdPoolLayer, self).get_layer_config(),
			"pool_size": self.pool_size,
			"strides": self.strides,
			"padding": self.padding
		}


class StdFlattenLayer(StdBaseLayer):
	layer_type = "Flatten"


class StdLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
		StdConvLayer,
		StdDenseLayer,
		StdPoolLayer,
		StdFlattenLayer
	]
	}


class StdLayersConfig(list):
	def __init__(self, basic_layers_config):
		for basic_layer in basic_layers_config:
			self.append(StdLayer.get(basic_layer.__class__.layer_type)(basic_layer))
	
	def get_config(self):
		return [layer.get_config() for layer in self]
	
	def get_init(self):
		return [layer.get_init() for layer in self]


class StdConfig:
	batch_size = None
	train_full = None
	epochs = None
	training_loop = None
	validation_loop = None
	scheme = None
	reg_type = None
	image_size = None
	layers = None
	minimize = None
	
	def __init__(self, snapshot):
		basic_config = BasicConfig(snapshot)
		self.batch_size = basic_config.batch_size
		self.train_full = basic_config.train_full
		self.epochs = basic_config.epochs
		self.training_loop = basic_config.training_loop
		self.validation_loop = basic_config.validation_loop
		self.scheme = basic_config.scheme
		self.reg_type = basic_config.reg_type
		self.image_size = basic_config.image_size
		self.layers = StdLayersConfig(basic_config.layers)
		self.minimize = basic_config.minimize
	
	def get_config(self):
		return {
			"batch_size": self.batch_size,
			"train_full": self.train_full,
			"epochs": self.epochs,
			"training_loop": self.training_loop,
			"validation_loop": self.validation_loop,
			"scheme": self.scheme,
			"reg_type": self.reg_type,
			"image_size": self.image_size,
			"layers": self.layers.get_config(),
			"minimize": self.minimize
		}
	
	def get_init(self):
		return self.layers.get_init()
	
	def dump(self, snapshot):
		with open(snapshot.config, "w") as f:
			json.dump(self.get_config(), f, indent='\t')
		with open(snapshot.init, "wb") as f:
			pkl.dump(self.get_init(), f)


# ---------- InferiorConfig ----------

class StdDividedConfig:
	_std_config = None
	
	def __init__(self, snapshot):
		self._std_config = StdConfig(snapshot)
	
	@property
	def std_config(self):
		return self._std_config
	
	@property
	def monitor_config(self):
		return MonitorConfig(self._std_config)
	
	@property
	def data_config(self):
		return DataConfig(self._std_config)
	
	@property
	def network_config(self):
		return NetworkConfig(self._std_config)
	
	def dump(self, snapshot):
		self.std_config.dump(snapshot)


class InferiorConfig:
	_std_config = None
	
	def __init__(self, std_config):
		self._std_config = std_config


class MonitorConfig(InferiorConfig):
	@property
	def training_loop(self):
		return self._std_config.training_loop
	
	@property
	def validation_loop(self):
		return self._std_config.validation_loop


class DataConfig(InferiorConfig):
	@property
	def batch_size(self):
		return self._std_config.batch_size
	
	@property
	def train_full(self):
		return self._std_config.train_full
	
	@property
	def epochs(self):
		return self._std_config.epochs
	
	@property
	def scheme(self):
		return self._std_config.scheme


class NetworkConfig(InferiorConfig):
	@property
	def reg_type(self):
		return self._std_config.reg_type
	
	@property
	def image_size(self):
		return self._std_config.image_size
	
	@property
	def layers(self):
		return self._std_config.layers
	
	@property
	def minimize(self):
		return self._std_config.minimize
