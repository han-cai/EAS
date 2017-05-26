from tf_network.rnn_seq import RNNSeq
import tensorflow as tf
import tf_network.tf_utils as TFUtils
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from util.config import add_domain2dict
from util.net_coder import NetCoder, NetStore
import numpy as np
from util.config import domain_key


class NASPolicyNet(RNNSeq):
	def build_graph(self):
		encoder_outputs, encoder_state, encoder_cell = self.build_encoder()
		self.update_ops["cell_state"] = encoder_state
		
		self._nas_one_step_sample(encoder_cell, domain="nas")
		nas_inputs = tf.stack(encoder_outputs, axis=0)  # (num_steps, batch_size, units)
		nas_inputs = tf.transpose(nas_inputs, [1, 0, 2])  # (batch_size, num_steps, units)
		self._nas_given_input_sample(nas_inputs, domain="nas")
		if self.config.get("build_train") is None:
			build_train = True
		else:
			build_train = self.config.get("build_train")
		if build_train:
			self._nas_train(nas_inputs, domain="nas")
	
	def _nas_classifier(self, cell_outputs, nas_config, reuse=False):
		dense_initializer = self.get_initializer(nas_config.get("dense_initializer"), "dense")
		output_initializer = self.get_initializer(nas_config.get("output_initializer"), "output")
		
		logits = TFUtils.classifier(cell_outputs, nas_config["output_dim"], nas_config["dense"], dense_initializer,
									output_initializer, reuse=reuse, scope="Output")  # (batch_size, n_out)
		if nas_config["output_dim"] > 1:
			probs = TFUtils.activation_func("softmax", logits)  # (batch_size, n_out)
		else:
			probs = TFUtils.activation_func("sigmoid", logits)  # (batch_size, n_out = 1)
			probs = tf.stack([1 - probs, probs], axis=1)  # (batch_size, 2)
		return probs
	
	def _nas_one_step_sample(self, cell, domain=None):
		encoder_config = self.config["encoder"]
		if encoder_config["rnn_type"] == "BiRNN":
			raise ValueError("NAS does not support BiRNN !")
		
		inputs, update_ops = {}, {}
		with tf.variable_scope(self.encoder_scope):
			# sample layer: first input
			if self.config.get("embedding") and self.config["embedding"]["use_embedding"]:
				inputs["one_step_token"] = tf.placeholder(tf.int32, [None], "one_step_token")  # (batch_size, )
				embedding_config = self.config["embedding"]
				embedding_initializer = self.get_initializer(embedding_config.get("initializer"))
				one_step_token, _ = TFUtils.embedding_layer(inputs["one_step_token"], embedding_config,
															embedding_initializer,
															reuse=True, scope=self.embed_scope)  # (batch_size, n_input)
			else:
				inputs["one_step_token"] = tf.placeholder(tf.float32, [None, self.config["input_dim"]],
														  "one_step_token")
				one_step_token = inputs["one_step_token"]  # (batch_size, n_input)
			# sample next layer: hidden states
			inputs["init_cell_state"] = tf.placeholder(tf.float32, [encoder_config["num_layers"], 2, None,
																	encoder_config["hidden_units"]], "init_cell_state")
			init_cell_state = tf.unstack(inputs["init_cell_state"], axis=0)
			init_cell_state = tuple([rnn.LSTMStateTuple(init_cell_state[_i][0], init_cell_state[_i][1])
									 for _i in range(encoder_config["num_layers"])])
			with tf.variable_scope(encoder_config["rnn_type"], reuse=True):
				cell_output, cell_state = cell(one_step_token, init_cell_state)
		
		# cell_output: (batch_size, units)
		nas_config = self.config["nas"]
		with tf.variable_scope("NAS"):
			probs = self._nas_classifier(cell_output, nas_config, reuse=False)  # (batch_size, act_space)
			update_ops["one_step_cell_state"] = cell_state
			update_ops["one_step_action_probs"] = probs
		self.inputs.update(add_domain2dict(inputs, domain))
		self.update_ops.update(add_domain2dict(update_ops, domain))
	
	def _nas_given_input_sample(self, nas_inputs, domain=None):
		# nas_inputs: (batch_size, num_steps, units)
		initializers, inputs, overheads, update_ops, summaries = {}, {}, {}, {}, {}
		
		seq_num = array_ops.shape(self.inputs["seq_len"])[0]
		gather_indices = tf.stack([tf.range(seq_num), self.inputs["seq_len"] - 1], 1)  # (batch_size, 2)
		gather_output = tf.gather_nd(nas_inputs, gather_indices)  # (batch_size, units)
		with tf.variable_scope("NAS"):
			probs = self._nas_classifier(gather_output, self.config["nas"], reuse=True)
			update_ops["given_input_action_probs"] = probs
		self.update_ops.update(add_domain2dict(update_ops, domain))
	
	def _nas_train(self, nas_inputs, domain=None):
		# nas_inputs: (batch_size, num_steps, units)
		initializers, inputs, overheads, update_ops, summaries = {}, {}, {}, {}, {}
		optimizer = TFUtils.build_optimizer(self.config["optimizer"])
		num_steps = self.config["encoder"]["num_steps"]
		
		nas_config = self.config["nas"]
		with tf.variable_scope("NAS"):
			inputs["reward"] = tf.placeholder(tf.float32, [None], "reward")  # (batch_size, )
			inputs["probs_mask"] = tf.placeholder(tf.float32, [None, num_steps, None], "probs_mask")
			inputs["action"] = tf.placeholder(tf.int32, [None, num_steps], "action")  # (batch_size, num_steps)
			inputs["action_start"] = tf.placeholder(tf.int32, [None], "action_start")  # (batch_size, )
			
			# flat_nas_inputs: (batch_size * num_steps, units)
			flat_nas_inputs = tf.reshape(nas_inputs, [-1, nas_inputs.shape.as_list()[2]])
			probs = self._nas_classifier(flat_nas_inputs, nas_config, reuse=True)  # (batch_size * num_steps, act_num)
			probs_mask = tf.reshape(inputs["probs_mask"], [-1, probs.shape.as_list()[1]])
			probs = tf.multiply(probs, probs_mask)  # (batch_size * num_steps, act_num)
			probs = tf.divide(probs, tf.reduce_sum(probs, axis=1, keep_dims=True))  # (batch_size * num_steps, act_num)
			
			indices = tf.reshape(inputs["action"], shape=[-1])  # (batch_size * num_steps, )
			indices = tf.stack([tf.range(array_ops.shape(indices)[0]), indices], axis=1)  # (batch_size * num_steps, 2)
			probs = tf.gather_nd(probs, indices)  # (batch_size * num_steps, )
			probs = tf.reshape(probs, shape=[-1, num_steps])  # (batch_size, num_steps)
			log_probs = tf.log(probs)  # (batch_size, num_steps)
			action_mask = TFUtils.matrix_mask(num_steps, inputs["action_start"], self.inputs["seq_len"] - 1)
			log_probs = tf.multiply(log_probs, action_mask)  # (batch_size, num_steps)
			obj = tf.multiply(log_probs, tf.expand_dims(inputs["reward"], axis=1))  # (batch_size, num_steps)
			obj = tf.reduce_sum(obj) / tf.cast(array_ops.shape(obj)[0], tf.float32)
			overheads["loss"] = -obj
			update_ops["reinforce"] = optimizer.minimize(overheads["loss"], self.overheads["global_step"])
		
		self.initializers.update(add_domain2dict(initializers, domain))
		self.inputs.update(add_domain2dict(inputs, domain))
		self.overheads.update(add_domain2dict(overheads, domain))
		self.update_ops.update(add_domain2dict(update_ops, domain))
		self.summaries.update(add_domain2dict(summaries, domain))
	
	def one_step_sample(self, one_step_token, init_cell_state, domain=None):
		action_probs, cell_state = self.sess.run([self.update_ops[domain_key("one_step_action_probs", domain)],
												  self.update_ops[domain_key("one_step_cell_state", domain)]],
												 feed_dict={
													 self.inputs[domain_key("one_step_token", domain)]: one_step_token,
													 self.inputs[domain_key("init_cell_state", domain)]: init_cell_state
												 })
		return action_probs, np.asarray(cell_state)
	
	def given_input_encode(self, net_seq, net_seq_len, domain=None):
		action_probs, cell_state = self.sess.run([self.update_ops[domain_key("given_input_action_probs", domain)],
												  self.update_ops["cell_state"]], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len
		})
		return action_probs, np.asarray(cell_state)
	
	def nas_reinforce(self, net_seq, net_seq_len, reward, probs_mask, action, action_start, domain=None):
		self.sess.run(self.update_ops[domain_key("reinforce", domain)], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs[domain_key("reward", domain)]: reward,
			self.inputs[domain_key("probs_mask", domain)]: probs_mask,
			self.inputs[domain_key("action", domain)]: action,
			self.inputs[domain_key("action_start", domain)]: action_start,
		})


class NASAgent:
	def __init__(self, config):
		model_config = config["agent_model_config"]
		net_coder_config = model_config["net_coder"]
		self.net_coder = NetCoder(net_coder_config)
		model_config["input_dim"] = self.net_coder.net_seq_dim
		model_config["nas"]["output_dim"] = self.net_coder.net_seq_dim
		self.agent_model = NASPolicyNet(model_config, config["model_restore_path"], config["exp_log_path"])
		
		if config["net_store_config"][0]:
			self.net_store_config = config["net_store_config"][1:]
			self.net_store = NetStore(config["net_store_config"][1])
		else:
			self.net_store = None
	
	def sample_nets(self, batch_size, layer_steps, net_config=None):
		num_steps = self.agent_model.config["encoder"]["num_steps"]
		
		store_actions = {"action": [], "probs_mask": [], "action_start": None}
		if net_config:
			# given input start
			net_configs = [[(name, layer_param.copy()) for name, layer_param in net_config] for _ in range(batch_size)]
			net_seq, net_seq_len = [0] * batch_size, [0] * batch_size
			for _i in range(batch_size):
				net_seq[_i], net_seq_len[_i] = self.net_coder.net_config2seq(net_configs[_i], num_steps)
			net_seq, net_seq_len = np.asarray(net_seq), np.asarray(net_seq_len)
			action_probs, cell_state = self.agent_model.given_input_encode(net_seq, net_seq_len, domain="nas")
			store_actions["action_start"] = net_seq_len - 1
		else:
			# empty start
			net_configs = [[] for _ in range(batch_size)]
			token = np.asarray([self.net_coder.one_hot[0] for _ in range(batch_size)])  # (batch_size, n_input)
			
			encoder_config = self.agent_model.config["encoder"]
			layer_num, units = encoder_config["num_layers"], encoder_config["hidden_units"]
			init_cell_state = np.zeros([layer_num, 2, batch_size, units])
			action_probs, cell_state = self.agent_model.one_step_sample(token, init_cell_state, domain="nas")
			store_actions["action_start"] = np.zeros([batch_size], dtype=np.int32)
		
		layer_types = self.net_coder.net_coder_config["layer_types"]
		for _t in range(batch_size):
			each_probs_mask = [np.ones([self.net_coder.net_seq_dim])] * store_actions["action_start"][_t]
			each_action = [0] * store_actions["action_start"][_t]
			each_act_prob = action_probs[_t:(_t + 1), :]
			each_cell_state = cell_state[:, :, _t:(_t + 1), :]
			for _s in range(layer_steps):
				if len(layer_types) > 1:
					valid_token = ["ltype_{}".format(ltype) for ltype in self.net_coder.net_coder_config["layer_types"]]
					valid_code = [self.net_coder.get_token_code(token) for token in valid_token]
					prob_mask = np.zeros([self.net_coder.net_seq_dim])
					for code in valid_code:
						prob_mask[code] = 1
					valid_probs = np.asarray([each_act_prob[0][code] for code in valid_code])
					valid_probs /= np.sum(valid_probs)
					action = int(np.random.multinomial(1, valid_probs).argmax())
					token, action = valid_token[action], valid_code[action]
					
					each_action.append(action)
					each_probs_mask.append(prob_mask)
					ltype = self.net_coder.get_code_token(action, "ltype").split("_")[1]
					token = self.net_coder.one_hot[self.net_coder.get_token_code(token)]
					each_act_prob, each_cell_state = \
						self.agent_model.one_step_sample(np.expand_dims(token, axis=0), each_cell_state, domain="nas")
				
				else:
					ltype = layer_types[0]
				
				param_dict = {}
				for param_name, param_vlist in self.net_coder.net_coder_config[ltype]:
					valid_token = ["{}&{}_{}".format(ltype, param_name, pval) for pval in param_vlist]
					valid_code = [self.net_coder.get_token_code(token) for token in valid_token]
					prob_mask = np.zeros([self.net_coder.net_seq_dim])
					for code in valid_code:
						prob_mask[code] = 1
					valid_probs = np.asarray([each_act_prob[0][code] for code in valid_code])
					valid_probs /= np.sum(valid_probs)
					action = int(np.random.multinomial(1, valid_probs).argmax())
					pval, token, action = param_vlist[action], valid_token[action], valid_code[action]
					
					each_action.append(action)
					each_probs_mask.append(prob_mask)
					param_dict[param_name] = pval
					token = self.net_coder.one_hot[self.net_coder.get_token_code(token)]
					each_act_prob, each_cell_state = \
						self.agent_model.one_step_sample(np.expand_dims(token, axis=0), each_cell_state, domain="nas")
				net_configs[_t].append((ltype, param_dict))
			each_action += [0] * (num_steps - len(each_action))
			each_probs_mask += [np.ones([self.net_coder.net_seq_dim])] * (num_steps - len(each_probs_mask))
			store_actions["action"].append(each_action)
			store_actions["probs_mask"].append(each_probs_mask)
		store_actions["action"] = np.asarray(store_actions["action"])
		store_actions["probs_mask"] = np.asarray(store_actions["probs_mask"])
		return store_actions, net_configs
	
	def update_agent(self, net_configs, actions, reward):
		num_steps = self.agent_model.config["encoder"]["num_steps"]
		action, probs_mask, action_start = actions["action"], actions["probs_mask"], actions["action_start"]
		
		net_seq, net_seq_len = [0] * len(net_configs), [0] * len(net_configs)
		for _i in range(len(net_configs)):
			net_seq[_i], net_seq_len[_i] = self.net_coder.net_config2seq(net_configs[_i], num_steps)
		net_seq, net_seq_len = np.asarray(net_seq), np.asarray(net_seq_len)
		self.agent_model.nas_reinforce(net_seq, net_seq_len, reward, probs_mask, action, action_start, domain="nas")
