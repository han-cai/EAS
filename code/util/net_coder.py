import pickle
import numpy as np
import os
import ast
from queue import PriorityQueue
from collections import deque


def vocab_list_2_vocab_map(vocab_list):
	vocab_map = {}
	_id = 0
	for token in vocab_list:
		vocab_map[token] = _id
		vocab_map[_id] = token
		_id += 1
	return vocab_map


class NetCoder:
	# net_config = [(layer_type, {param_name: param_val, ...}), ...]
	# net_str (as key)
	# net_code, net_seq
	def __init__(self, net_coder_config):
		self.net_coder_config = net_coder_config
		self.net_seq_dim = 0
		self.param_dim = 0
		self.vocab = self._build_vocab()
		self.one_hot = np.identity(self.net_seq_dim, dtype=np.float32)
	
	def _build_vocab(self):
		vocab = {}
		layer_types = self.net_coder_config["layer_types"]
		for ltype in layer_types:
			for param_name, param_vlist in self.net_coder_config[ltype]:
				self.param_dim = max(self.param_dim, len(param_vlist))
		if self.net_coder_config["scheme"]["token_per_layer"]:
			vocab_list = []
			for ltype in layer_types:
				layer_token = deque()
				layer_token.append(ltype)
				for param_name, param_vlist in self.net_coder_config[ltype]:
					qsize = len(layer_token)
					for _ in range(qsize):
						token = layer_token.popleft()
						for val in param_vlist:
							layer_token.append(token + "&{}_{}".format(param_name, val))
				vocab_list += list(layer_token)
			vocab = vocab_list_2_vocab_map(vocab_list)
			self.net_seq_dim = len(vocab_list)
		elif self.net_coder_config["scheme"]["separate_vocab"]:
			if len(layer_types) > 1:
				vocab["ltype"] = vocab_list_2_vocab_map(["ltype_{}".format(ltype) for ltype in layer_types])
				self.net_seq_dim = max(self.net_seq_dim, len(layer_types))
			for ltype in layer_types:
				for param_name, param_vlist in self.net_coder_config[ltype]:
					vocab_key = "{}&{}".format(ltype, param_name)
					vocab_list = ["{}_{}".format(vocab_key, param_val) for param_val in param_vlist]
					vocab[vocab_key] = vocab_list_2_vocab_map(vocab_list)
					self.net_seq_dim = max(self.net_seq_dim, len(vocab_list))
		else:
			vocab_list = []
			if len(layer_types) > 1: vocab_list.extend(["ltype_{}".format(ltype) for ltype in layer_types])
			for ltype in layer_types:
				for param_name, param_vlist in self.net_coder_config[ltype]:
					vocab_key = "{}&{}".format(ltype, param_name)
					vocab_list.extend(["{}_{}".format(vocab_key, param_val) for param_val in param_vlist])
			vocab = vocab_list_2_vocab_map(vocab_list)
			self.net_seq_dim = len(vocab_list)
		return vocab
	
	def net_config2str(self, net_config):
		net_tokens = []
		for ltype, layer_params in net_config:
			layer_token = ltype
			for param_name, _ in self.net_coder_config[ltype]:
				layer_token += "&{}_{}".format(param_name, layer_params[param_name])
			net_tokens.append(layer_token)
		return " ".join(net_tokens)
	
	def net_config2code(self, net_config):
		token_list = []
		for layer_id in range(len(net_config)):
			ltype, layer_params = net_config[layer_id]
			if self.net_coder_config["scheme"]["token_per_layer"]:
				layer_token = ltype
				for param_name, _ in self.net_coder_config[ltype]:
					layer_token += "&{}_{}".format(param_name, layer_params[param_name])
				token_list.append(layer_token)
			else:
				if len(self.net_coder_config["layer_types"]) > 1:
					token_list.append("ltype_{}".format(ltype))
				for param_name, _ in self.net_coder_config[ltype]:
					token_list.append("{}&{}_{}".format(ltype, param_name, layer_params[param_name]))
		net_code = []
		for token in token_list:
			net_code.append(self.get_token_code(token))
		return net_code, token_list
	
	def net_code2seq(self, net_code, num_steps):
		net_seq = np.zeros((num_steps, self.net_seq_dim))
		net_seq_len = len(net_code)
		for _i in range(min(net_seq_len, num_steps)):
			net_seq[_i] = self.one_hot[net_code[_i]]
		return net_seq, net_seq_len
	
	def net_config2seq(self, net_config, num_steps):
		net_code, token_list = self.net_config2code(net_config)
		net_seq, net_seq_len = self.net_code2seq(net_code, num_steps)
		return net_seq, net_seq_len
	
	@staticmethod
	def net_seq2code(net_seq, net_seq_len):
		net_code = []
		for _t in range(net_seq_len):
			net_code.append(np.argmax(net_seq[_t]))
		return net_code
	
	def net_code2config(self, net_code):
		net_config = []
		if self.net_coder_config["scheme"]["token_per_layer"]:
			for code in net_code:
				ltype, layer_params = self.vocab[code].split("&")
				param_dict = {}
				for param in layer_params:
					param_name, param_val = param.split("_")
					param_dict[param_name] = ast.literal_eval(param_val)
				net_config.append((ltype, param_dict))
		else:
			code_queue = deque(net_code)
			while len(code_queue) != 0:
				if len(self.net_coder_config["layer_types"]) == 1:
					ltype = self.net_coder_config["layer_types"][0]
				else:
					layer_code = code_queue.popleft()
					ltype = self.get_code_token(layer_code, "ltype").split("_")[1]
				param_dict = {}
				for param_name, param_vlist in self.net_coder_config[ltype]:
					param_code = code_queue.popleft()
					param_val = self.get_code_token(param_code, "{}&{}".format(ltype, param_name)).split("_")[1]
					param_dict[param_name] = ast.literal_eval(param_val)
				net_config.append((ltype, param_dict))
		return net_config
	
	def net_seq2config(self, net_seq, net_seq_len):
		net_code = self.net_seq2code(net_seq, net_seq_len)
		net_config = self.net_code2config(net_code)
		return net_config
	
	def get_param_vlist(self, ltype, param_name):
		params = self.net_coder_config[ltype]
		for name, vlist in params:
			if name == param_name:
				return vlist
		raise KeyError("{} {} not found.".format(ltype, param_name))
	
	def get_token_code(self, token):
		if type(self.vocab[list(self.vocab.keys())[0]]) == dict:
			vocab_key = token.split("_")[0]
			code = self.vocab[vocab_key][token]
		else:
			code = self.vocab[token]
		return code
	
	def get_code_token(self, code, domain):
		if type(self.vocab[list(self.vocab.keys())[0]]) == dict:
			return self.vocab[domain][code]
		else:
			return self.vocab[code]
	
	def sample_net_configs(self, layer_num, batch_size, same=False, min_units=False):
		if batch_size == 1:
			net_config = []
			for _i in range(layer_num):
				layer_types = self.net_coder_config["layer_types"]
				ltype = layer_types[np.random.randint(0, len(layer_types))]
				param_dict = {}
				for param_name, param_vlist in self.net_coder_config[ltype]:
					if min_units and ((ltype == "C" and param_name == "FN")
									  or (ltype == "FC" and param_name == "units")):
						pval = param_vlist[0]
					else:
						pval = param_vlist[np.random.randint(0, len(param_vlist))]
					param_dict[param_name] = pval
				net_config.append((ltype, param_dict))
			return net_config
		elif same:
			net_config = self.sample_net_configs(layer_num, 1, same, min_units)
			return [[(name, layer_param.copy()) for name, layer_param in net_config] for _ in range(batch_size)]
		else:
			return [self.sample_net_configs(layer_num, 1, same, min_units) for _ in range(batch_size)]
	
	def widen_valid_layer(self, net_config):
		token_list = []
		valid_action, valid_layer = [], []
		for layer_id in range(len(net_config)):
			ltype, layer_params = net_config[layer_id]
			if self.net_coder_config["scheme"]["token_per_layer"]:
				layer_token = ltype
				for param_name, _ in self.net_coder_config[ltype]:
					layer_token += "&{}_{}".format(param_name, layer_params[param_name])
				token_list.append(layer_token)
			else:
				if len(self.net_coder_config["layer_types"]) > 1:
					token_list.append("ltype_{}".format(ltype))
				for param_name, _ in self.net_coder_config[ltype]:
					token_list.append("{}&{}_{}".format(ltype, param_name, layer_params[param_name]))
			if (ltype == "C" and layer_params["FN"] < self.get_param_vlist("C", "FN")[-1]) or \
					(ltype == "FC" and layer_params["units"] < self.get_param_vlist("FC", "units")[-1]):
				valid_action.append(len(token_list) - 1)
				valid_layer.append(layer_id)
		return valid_action, valid_layer
	
	def widen(self, layer_config):
		ltype, layer_param = layer_config[0], layer_config[1].copy()
		if ltype == "C":
			vlist = self.get_param_vlist("C", "FN")
			layer_param["FN"] = vlist[vlist.index(layer_param["FN"]) + 1]
		elif ltype == "FC":
			vlist = self.get_param_vlist("FC", "units")
			layer_param["units"] = vlist[vlist.index(layer_param["units"]) + 1]
		else:
			raise ValueError("Not able to perform widen operation to {}".format(ltype))
		return ltype, layer_param
	
	@staticmethod
	def deepen_valid_place(net_config):
		return np.arange(len(net_config))


class NetStore:
	def __init__(self, restore_path=None):
		if restore_path:
			try:
				with open(restore_path, "rb") as fin:
					net_dict = pickle.load(fin)
					self.net_value_map = net_dict["value"]
					self.net_weight_map = net_dict["weight"]
				print("restore from {}".format(restore_path))
			except:
				print("Path {} not exit.".format(restore_path))
				self.net_value_map, self.net_weight_map = {}, {}
		else:
			self.net_value_map, self.net_weight_map = {}, {}
	
	def add_net_value(self, net_str, value):
		value_list = self.net_value_map.get(net_str)
		if not value_list:
			value_list = []
		value_list.append(value)
		self.net_value_map[net_str] = value_list
	
	def statistics(self):
		vals = []
		for net_str in self.net_value_map.keys():
			value_list = self.net_value_map[net_str]
			val = np.mean(value_list)
			vals.append(val)
		return len(vals), np.max(vals)
	
	def get_net_value(self, net_str):
		return self.net_value_map.get(net_str)
	
	def add_net_weights(self, net_str, weight, value, max_weight_store_per_net=1):
		weight_queue = self.net_weight_map.get(net_str)
		if weight_queue is None:
			weight_queue = PriorityQueue()
		if weight_queue.qsize() < max_weight_store_per_net:
			weight_queue.put((value, weight))
		else:
			v, w = weight_queue.queue[0]
			if value > v:
				weight_queue.get()
				weight_queue.put((value, weight))
	
	def get_net_weights(self, net_str, scheme="max"):
		weight_queue = self.net_weight_map.get(net_str)
		if weight_queue:
			if scheme == "max":
				value, weights = weight_queue.queue[0]
				for _i in range(1, weight_queue.qsize()):
					v, w = weight_queue.queue[_i]
					if v > value: value, weights = v, w
				return value, weights
			else:
				# random
				index = np.random.randint(0, weight_queue.qsize())
				return weight_queue.queue[index]
		else:
			return None, None
	
	def save(self, save_path):
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		net_store = {"value": self.net_value_map, "weight": self.net_weight_map}
		pickle.dump(net_store, open(save_path + "net.store", "wb"))


class Population:
	def __init__(self, net_coder, max_size=50):
		self.population = PriorityQueue(max_size)
		self.net_coder = net_coder
	
	def add(self, net_val, net_config):
		if self.population.full():
			val, config = self.population.queue[0]
			if net_val > val:
				self.population.get()
				self.population.put((net_val, net_config))
		else:
			self.population.put((net_val, net_config))
	
	def sample_net(self, batch_size=None):
		if batch_size is None:
			index = np.random.randint(0, self.population.qsize())
			val, config = self.population.queue[index]
			return val, config
		else:
			index = np.random.randint(0, self.population.qsize(), batch_size).tolist()
			net_configs = []
			net_vals = []
			for i in index:
				val, config = self.population.queue[i]
				net_vals.append(val)
				net_configs.append(config)
			return net_vals, net_configs
	
	def get_max_val(self):
		if self.population.qsize() == 0:
			return 0, None
		value, weights = self.population.queue[0]
		for _i in range(1, self.population.qsize()):
			v, w = self.population.queue[_i]
			if v > value: value, weights = v, w
		return value, weights

	def get_size(self):
		return self.population.qsize()
