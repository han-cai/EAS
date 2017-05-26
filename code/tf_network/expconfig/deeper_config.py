import numpy as np
from tf_network.tf_utils import KeyMap

class DeeperBaseLayer:
	std_layer = None

	def __init__(self, std_layer):
		self.std_layer = std_layer

class DeeperLinearLayer(DeeperBaseLayer):
	def identity(self, kwargs, noise=None):
		new_std_layer = self.std_layer.__class__(self.std_layer)

		self.identity_modification(new_std_layer, kwargs)
		if noise:
			noise *= np.sqrt(2.0 / np.prod(new_std_layer.init["kernel"].shape[:-1]))
			new_std_layer.init["kernel"] += np.random.normal(scale=noise, size=new_std_layer.init["kernel"].shape)
		new_std_layer.init_bias = 0
		new_std_layer.keep_prob = 1
		new_std_layer.use_bn = True
		if new_std_layer.use_bn:
			new_std_layer.init["scale"] = np.ones(new_std_layer.init["kernel"].shape[-1])
			new_std_layer.init["offset"] = np.zeros(new_std_layer.init["kernel"].shape[-1])

		return new_std_layer

	def identity_modification(self, new_std_layer, kwargs):
		pass

class DeeperConvLayer(DeeperLinearLayer):
	layer_type = "Conv"

	@property
	def filters(self):
		return self.std_layer.filters

	@property
	def kernel_size(self):
		return self.std_layer.kernel_size

	def identity_modification(self, new_std_layer, kwargs):
		assert kwargs.get("filters", self.filters) == self.filters, "Filters can not be changed."
		new_kernel_size = kwargs.get("kernel_size", self.kernel_size)

		new_std_layer.init = {
			"kernel": self.identity_kernel(new_kernel_size)
		}
		new_std_layer.kernel_size = new_kernel_size

	def identity_kernel(self, new_kernel_size):
		mid = new_kernel_size // 2
		kernel = np.zeros([new_kernel_size, new_kernel_size, self.filters, self.filters])
		kernel[mid, mid] = np.eye(self.filters)
		return kernel

class DeeperDenseLayer(DeeperLinearLayer):
	layer_type = "Dense"

	@property
	def units(self):
		return self.std_layer.units

	def identity_modification(self, new_std_layer, kwargs):
		assert kwargs.get("units", self.units) == self.units, "Units can not be changed."

		new_std_layer.init = {
			"kernel": self.identity_kernel()
		}

	def identity_kernel(self):
		return np.eye(self.units)

class DeeperPoolLayer(DeeperBaseLayer):
	layer_type = "Pool"

class DeeperFlattenLayer(DeeperBaseLayer):
	layer_type = "Flatten"

class DeeperLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
			DeeperConvLayer,
			DeeperDenseLayer,
			DeeperPoolLayer,
			DeeperFlattenLayer
		]
	}

class DeeperLayersConfig(list):
	std_layers_config = None

	def __init__(self, std_layers_config):
		self.std_layers_config = std_layers_config
		for std_layer in std_layers_config:
			self.append(DeeperLayer.get(std_layer.__class__.layer_type)(std_layer))

	@property
	def linear_idx_list(self):
		return [i for i, layer in enumerate(self) if isinstance(layer, DeeperLinearLayer)]

	def deepen(self, idx, kwargs, noise=None):
		assert not isinstance(self[idx], DeeperFlattenLayer), "can't insert a layer after flatten layer"
		
		pre_idx = idx
		while not isinstance(self[pre_idx], DeeperLinearLayer):
			pre_idx -= 1
		self.std_layers_config.insert(idx + 1, self[pre_idx].identity(kwargs, noise))

class DeeperNetworkConfig:
	layers = None

	def __init__(self, std_network_config):
		self.layers = DeeperLayersConfig(std_network_config.layers)

	def apply(self, idx, kwargs, noise=None):
		return self.layers.deepen(idx, kwargs, noise)
