import numpy as np
from util.net_coder import NetCoder
from tf_network.rnn_seq2v import RNNSeq2V


class NetValuePredictor:
	def __init__(self, config):
		model_config = config["model_config"]
		if config["model"] == "RNNSeq2V":
			net_coder_config = model_config["net_coder"]
			self.net_coder = NetCoder(net_coder_config)
			model_config["input_dim"] = self.net_coder.net_seq_dim
			self.model = RNNSeq2V(model_config, config["model_restore_path"], config["exp_log_path"])
			print("A RNNSeq2V model is used.")
		else:
			self.model = None
			print("A default score model (hash function) is used.")
	
	@staticmethod
	def _default_score_function(net_str):
		hash_value = hash(net_str)
		hash_value %= 5
		hash_value -= 2.5
		return np.tanh(hash_value)
	
	def get_net_value(self, batch_net_config):
		if self.model:
			num_steps = self.model.config["encoder"]["num_steps"]
			net_seq, net_seq_len = [], []
			for net_config in batch_net_config:
				net_code, _ = self.net_coder.net_config2code(net_config)
				seq, seq_len = self.net_coder.net_code2seq(net_code, num_steps)
				net_seq.append(seq)
				net_seq_len.append(seq_len)
			net_seq = np.stack(net_seq, axis=0)
			net_seq_len = np.asarray(net_seq_len)
			pVals = self.model.seq2v_query(net_seq, net_seq_len)
			return pVals[:, 0]
		else:
			batch_net_value = []
			for net_config in batch_net_config:
				batch_net_value.append(self._default_score_function(self.net_coder.net_config2str(net_config)))
			return np.asarray(batch_net_value)


class Oracle(NetValuePredictor):
	def __init__(self, config):
		NetValuePredictor.__init__(self, config)
		self.size_penalty_scheme = config["size_penalty_scheme"]
	
	def _calc_size_penalty(self, batch_net_config):
		if self.size_penalty_scheme == "none":
			return np.zeros(len(batch_net_config))
		else:
			pass  # TODO
	
	def get_score(self, batch_net_config):
		size_penalty = self._calc_size_penalty(batch_net_config)
		performance_score = self.get_net_value(batch_net_config)
		return performance_score - size_penalty


class VirtualEnv:
	def __init__(self, oracle_config):
		self.oracle = Oracle(oracle_config)
	
	def reward(self, batch_net_config):
		return self.oracle.get_score(batch_net_config)
