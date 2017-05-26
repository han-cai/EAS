import json
import pickle as pkl
from os.path import exists
from tf_network.layers import KeyMap


class BasicBaseLayer:
	_layer_config = None
	_init = None
	
	def __init__(self, layer_config, init_config):
		self._layer_config = layer_config
		self._init = init_config
	
	@property
	def keep_prob(self):
		return self._layer_config.get("keep_prob", None)
	
	@property
	def init(self):
		return self._init


class BasicLinearLayer(BasicBaseLayer):
	@property
	def stddev(self):
		stddev = self._layer_config.get("stddev", ["hekaiming", {}])
		if isinstance(stddev, float):
			return ["truncated_normal", {"stddev": stddev}]
		elif isinstance(stddev, str):
			return [stddev, {}]
		else:
			init_type, init_args = stddev
			if isinstance(init_args, float):
				return [init_type, {"stddev": init_args}]
			else:
				return stddev
	
	@property
	def weight_decay(self):
		weight_decay = self._layer_config.get("weight_decay", [None, None])
		if isinstance(weight_decay, float):
			return ["l2", weight_decay]
		else:
			return weight_decay
	
	@property
	def activation(self):
		return self._layer_config.get("activation", None)
	
	@property
	def use_bn(self):
		return self._layer_config.get("use_bn", False)
	
	@property
	def init_bias(self):
		return self._layer_config.get("init_bias", 0)


class BasicConvLayer(BasicLinearLayer):
	layer_type = "Conv"
	
	@property
	def filters(self):
		return self._layer_config["filters"]
	
	@property
	def kernel_size(self):
		return self._layer_config["kernel_size"]
	
	@property
	def padding(self):
		return self._layer_config.get("padding", "same")


class BasicDenseLayer(BasicLinearLayer):
	layer_type = "Dense"
	
	@property
	def units(self):
		return self._layer_config["units"]


class BasicPoolLayer(BasicBaseLayer):
	layer_type = "Pool"
	
	@property
	def pool_size(self):
		return self._layer_config["pool_size"]
	
	@property
	def strides(self):
		return self._layer_config["strides"]
	
	@property
	def padding(self):
		return self._layer_config.get("padding", "same")


class BasicFlattenLayer(BasicBaseLayer):
	layer_type = "Flatten"


class BasicLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
			BasicConvLayer,
			BasicDenseLayer,
			BasicPoolLayer,
			BasicFlattenLayer
		]
	}


class BasicLayersConfig(list):
	def __init__(self, layers, init):
		init = init or [{} for _ in layers]
		for layer, init_config in zip(layers, init):
			layer_type, layer_config = layer
			self.append(BasicLayer.get(layer_type)(layer_config, init_config))


class BasicConfig:
	_config = None
	_init = None
	
	def __init__(self, snapshot):
		with open(snapshot.config, "r") as f:
			self._config = json.load(f)
		if exists(snapshot.init):
			with open(snapshot.init, "rb") as f:
				self._init = pkl.load(f)
	
	@property
	def batch_size(self):
		return self._config["batch_size"]
	
	@property
	def train_full(self):
		return self._config["train_full"]
	
	@property
	def epochs(self):
		return self._config["epochs"]
	
	@property
	def training_loop(self):
		return self._config["training_loop"]
	
	@property
	def validation_loop(self):
		return self._config["validation_loop"]
	
	@property
	def scheme(self):
		return self._config["scheme"]
	
	@property
	def reg_type(self):
		return self._config["reg_type"]
	
	@property
	def image_size(self):
		return self._config["image_size"]
	
	@property
	def layers(self):
		return BasicLayersConfig(self._config["layers"], self._init)
	
	@property
	def minimize(self):
		return self._config["minimize"]
