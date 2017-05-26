from tf_network.layers import KeyMap


class RenewBaseLayer:
	runtime_layer = None
	std_layer = None
	graph = None
	scope = None
	index = None
	
	def __init__(self, runtime_layer, std_layer, graph, scope, index):
		self.runtime_layer = runtime_layer
		self.std_layer = std_layer
		self.graph = graph
		self.scope = scope
		self.index = index
	
	@property
	def init(self):
		assert False, "Do not try to fetch init."
	
	@init.setter
	def init(self, val):
		self.std_layer.init = val
	
	def renew(self, sess):
		self.init = {
			k: sess.run(self.get_tensor(v)) for k, v in self.names_map.items()
		}
	
	@property
	def name(self):
		return self.runtime_layer.name(self.index)
	
	def get_tensor(self, v):
		return self.graph.get_tensor_by_name(self.get_tensor_name(v))
	
	def get_tensor_name(self, v):
		return "{}/{}/{}:0".format(self.scope, self.name, v)
	
	@property
	def names_map(self):
		return {}


class RenewLinearLayer(RenewBaseLayer):
	@property
	def linear_name(self):
		return self.runtime_layer.linear_name
	
	@property
	def bn_name(self):
		return self.runtime_layer.bn_name
	
	@property
	def kernel_name(self):
		return "{}/{}".format(self.linear_name, "kernel")
	
	@property
	def bias_name(self):
		return "{}/{}".format(self.linear_name, "bias")
	
	@property
	def scale_name(self):
		return "{}/{}".format(self.bn_name, "scale")
	
	@property
	def offset_name(self):
		return "{}/{}".format(self.bn_name, "offset")
	
	@property
	def use_bn(self):
		return self.runtime_layer.use_bn
	
	@property
	def names_map(self):
		m = {
			"kernel": self.kernel_name
		}
		if self.use_bn:
			m.update({
				"scale": self.scale_name,
				"offset": self.offset_name
			})
		else:
			m.update({
				"bias": self.bias_name
			})
		return m


class RenewConvLayer(RenewLinearLayer):
	layer_type = "Conv"


class RenewDenseLayer(RenewLinearLayer):
	layer_type = "Dense"


class RenewPoolLayer(RenewBaseLayer):
	layer_type = "Pool"


class RenewFlattenLayer(RenewBaseLayer):
	layer_type = "Flatten"


class RenewLayer(KeyMap):
	_map = {
		layer_class.layer_type: layer_class
		for layer_class in [
			RenewConvLayer,
			RenewDenseLayer,
			RenewPoolLayer,
			RenewFlattenLayer
		]
	}


class RenewLayersConfig(list):
	def __init__(self, runtime_layers_config, std_layers_config, graph, scope):
		for i, (runtime_layer, std_layer) in enumerate(zip(runtime_layers_config, std_layers_config)):
			self.append(RenewLayer.get(std_layer.__class__.layer_type)(runtime_layer, std_layer, graph, scope, i))
	
	def renew(self, sess):
		for renew_layer in self:
			renew_layer.renew(sess)


# ---------- RenewNetworkConfig ----------

class RenewNetworkConfig:
	layers = None
	
	def __init__(self, runtime_network_config, network_config, graph, scope):
		self.layers = RenewLayersConfig(runtime_network_config.layers, network_config.layers, graph, scope)
	
	def apply(self, sess):
		self.layers.renew(sess)
