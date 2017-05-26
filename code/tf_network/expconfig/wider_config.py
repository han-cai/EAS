import numpy as np
from tf_network.layers import KeyMap

class Expander:
	@staticmethod
	def make_indices(size, new_size):
		base = np.arange(size)
		return np.concatenate([base, np.random.choice(base, new_size - size)])

	@staticmethod
	def make_magnifier(size, indices):
		l = np.zeros(size)
		for x in indices:
			l[x] += 1
		return (1.0 / l)[indices]

	@staticmethod
	def sample(size, new_size):
		indices = Expander.make_indices(size, new_size)
		magnifier = Expander.make_magnifier(size, indices)
		return indices, magnifier

class ValidSizes(list):
	def get_next(self, current_value):
		for x in self:
			if x > current_value:
				return x
		return current_value

class WiderBaseLayer:
	_std_layer = None

	def __init__(self, std_layer):
		self._std_layer = std_layer

class WiderLinearLayer(WiderBaseLayer):
	@property
	def kernel(self):
		return self._std_layer.init["kernel"]

	@kernel.setter
	def kernel(self, val):
		self._std_layer.init["kernel"] = val

	@property
	def bias(self):
		return self._std_layer.init["bias"]

	@bias.setter
	def bias(self, val):
		self._std_layer.init["bias"] = val

	@property
	def use_bn(self):
		return self._std_layer.use_bn

	@property
	def scale(self):
		return self._std_layer.init["scale"]

	@scale.setter
	def scale(self, val):
		self._std_layer.init["scale"] = val

	@property
	def offset(self):
		return self._std_layer.init["offset"]

	@offset.setter
	def offset(self, val):
		self._std_layer.init["offset"] = val

	# widen

	def widen(self, noise=None, drop=1):
		size, new_size = self.size, self.get_new_size()
		indices, magnifier = Expander.sample(size, new_size)
		self.widen_kernel(indices, magnifier)
		if self.use_bn:
			self.widen_scale(indices, magnifier)
			self.widen_offset(indices, magnifier)
		else:
			self.widen_bias(indices, magnifier)
		if self._std_layer.keep_prob is None:
			self._std_layer.keep_prob = drop
		if noise:
			if len(self.kernel.shape) == 2:
				weight_noise = noise * np.sqrt(2.0 / np.prod(self.kernel.shape[:-1]))
				self.kernel[:, size:] += np.random.normal(scale=weight_noise, size=self.kernel[:, size:].shape)
			else:
				weight_noise = noise * np.sqrt(2.0 / np.prod(self.kernel.shape[:-1]))
				self.kernel[:, :, :, size:] += np.random.normal(scale=weight_noise, size=self.kernel[:, :, :, size:].shape)
		self.size = new_size
		return indices, magnifier, size

	def get_new_size(self):
		return self.__class__.valid_sizes.get_next(self.size)

	def widen_kernel(self, indices, magnifier):
		pass

	def widen_bias(self, indices, magnifier):
		# self.bias = self.bias[indices] * magnifier
		self.bias = self.bias[indices]

	def widen_scale(self, indices, magnifier):
		self.scale = self.scale[indices]

	def widen_offset(self, indices, magnifier):
		# self.offset = self.offset[indices] * magnifier
		self.offset = self.offset[indices]

	# prev_widen

	def prev_widen(self, origin, indices, magnifier, prev_size):
		pass


class WiderConvLayer(WiderLinearLayer):
	layer_type = "Conv"
	valid_sizes = ValidSizes([32, 48, 64, 96, 128, 192, 256, 384, 512])

	@property
	def size(self):
		return self._std_layer.filters

	@size.setter
	def size(self, val):
		self._std_layer.filters = val

	# widen

	def widen_kernel(self, indices, magnifier):
		# self.kernel = self.kernel[:, :, :, indices] * magnifier
		self.kernel = self.kernel[:, :, :, indices]

	# prev_widen

	def prev_widen(self, origin, indices, magnifier, prev_size):
		# self.kernel = self.kernel[:, :, indices, :]
		self.kernel = self.kernel[:, :, indices, :] * magnifier.reshape([1, 1, -1, 1])

class WiderDenseLayer(WiderLinearLayer):
	layer_type = "Dense"
	valid_sizes = ValidSizes([64, 96, 128, 192, 256, 320, 416, 512, 768, 1024])

	@property
	def size(self):
		return self._std_layer.units

	@size.setter
	def size(self, val):
		self._std_layer.units = val

	# widen

	def widen_kernel(self, indices, magnifier):
		# self.kernel = self.kernel[:, indices] * magnifier
		self.kernel = self.kernel[:, indices]

	# prev_widen

	def prev_widen(self, origin, indices, magnifier, prev_size):
		if isinstance(origin, WiderDenseLayer):
			# self.kernel = self.kernel[indices, :]
			self.kernel = self.kernel[indices, :] * magnifier.reshape([-1, 1])
		elif isinstance(origin, WiderConvLayer):
			kernel = self.kernel.reshape((-1, prev_size, self.size))
			# self.kernel = kernel[:, indices, :].reshape((-1, self.size))
			self.kernel = (kernel[:, indices, :] * magnifier.reshape([1, -1, 1])).reshape((-1, self.size))
		else:
			assert False, "Unrecognizable WiderLayer type {}.".format(origin.__class__)

class WiderPoolLayer(WiderBaseLayer):
	layer_type = "Pool"

class WiderFlattenLayer(WiderBaseLayer):
	layer_type = "Flatten"

class WiderLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
			WiderConvLayer,
			WiderDenseLayer,
			WiderPoolLayer,
			WiderFlattenLayer
		]
	}

class WiderLayersConfig(list):
	def __init__(self, std_layers_config):
		for std_layer in std_layers_config:
			self.append(WiderLayer.get(std_layer.__class__.layer_type)(std_layer))

	@property
	def linear_idx_list(self):
		return [i for i, layer in enumerate(self) if isinstance(layer, WiderLinearLayer)]

	def widen(self, binary_list, noise=None, drop=1):
		linear_idx_list = self.linear_idx_list
		for idx, next_idx, flag in zip(linear_idx_list[:-1], linear_idx_list[1:], binary_list):
			if flag:
				origin = self[idx]
				indices, magnifier, prev_size = origin.widen(noise, drop)
				self[next_idx].prev_widen(origin, indices, magnifier, prev_size)

class WiderNetworkConfig:
	layers = None

	def __init__(self, std_network_config):
		self.layers = WiderLayersConfig(std_network_config.layers)

	def apply(self, binary_list, noise=None, drop=1):
		return self.layers.widen(binary_list, noise, drop)
