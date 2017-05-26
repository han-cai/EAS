from util.net_coder import NetCoder, NetStore
import tensorflow as tf
import tf_network.tf_utils as TFUtils
from tensorflow.python.ops import array_ops
import numpy as np
from tf_network.rnn_seq import RNNSeq
from util.config import make_matrix_mask
from util.config import add_domain2dict, domain_key


class Net2NetPolicyNet(RNNSeq):
	def build_graph(self):
		encoder_outputs, _, _ = self.build_encoder()
		
		states = tf.stack(encoder_outputs, axis=0)  # (num_steps, states_num, units)
		states = tf.transpose(states, [1, 0, 2])  # (states_num, num_steps, units)
		
		self.inputs["state_seg"] = tf.placeholder(tf.int32, shape=(), name="state_seg")
		n2w_states = states[:self.inputs["state_seg"]]  # (n2w_states_num, num_steps, units)
		n2d_states = states[self.inputs["state_seg"]:]  # (n2d_states_num, num_steps, units)
		
		if self.config.get("mode") is None:
			n2w_build_train, n2d_build_train, n2n_build_train = False, False, False
		elif self.config["mode"] == "net2wider":
			n2w_build_train, n2d_build_train, n2n_build_train = True, False, False
		elif self.config["mode"] == "net2deeper":
			n2w_build_train, n2d_build_train, n2n_build_train = False, True, False
		else:
			n2w_build_train, n2d_build_train, n2n_build_train = False, False, True
		
		n2w_obj = self.net2wider_decoder(n2w_states, "net2wider", build_train=n2w_build_train)
		n2d_obj = self.net2deeper_decoder(n2d_states, "net2deeper", build_train=n2d_build_train)
		if n2n_build_train:
			optimizer = TFUtils.build_optimizer(self.config["optimizer"])
			domain = "net2net"
			with tf.variable_scope("Net2Net"):
				n2n_obj = n2w_obj + n2d_obj
				self.inputs[domain_key("episode_num", domain)] = tf.placeholder(tf.float32, (), name="episode_num")
				self.overheads[domain_key("loss", domain)] = -n2n_obj / self.inputs[domain_key("episode_num", domain)]
				self.update_ops[domain_key("reinforce", domain)] = \
					optimizer.minimize(self.overheads[domain_key("loss", domain)], self.overheads["global_step"])
	
	def net2wider_decoder(self, states, domain=None, build_train=True):
		# states: (states_num, num_steps, units)
		initializers, inputs, overheads, update_ops, summaries = {}, {}, {}, {}, {}
		optimizer = TFUtils.build_optimizer(self.config["optimizer"])
		states_num = array_ops.shape(states)[0]
		
		config = self.config["net2wider"]
		with tf.variable_scope("Net2Wider"):
			# sample action
			with tf.variable_scope("Sample"):
				# input valid_action: (states_num, max_ac_num)
				inputs["valid_action"] = tf.placeholder(tf.int32, shape=[None, None], name="valid_action")
				tmp = tf.expand_dims(tf.range(states_num), axis=1) + \
					  tf.zeros_like(inputs["valid_action"], dtype=tf.int32)  # (states_num, max_ac_num)
				indices = tf.stack([tmp, inputs["valid_action"]], axis=2)  # (states_num, max_ac_num, 2)
				indices = tf.reshape(indices, [-1, 2])  # (states_num * max_ac_num, 2)
				states = tf.gather_nd(states, indices)  # (states_num * max_ac_num, units)
				if config["seq2seq"]:
					pass  # TODO
				else:
					dense_initializer = self.get_initializer(config.get("dense_initializer"), "dense")
					output_initializer = self.get_initializer(config.get("output_initializer"), "output")
					logits = TFUtils.classifier(states, 1, config["dense"], dense_initializer, output_initializer,
												reuse=False, scope="classifier")  # (states_num * max_ac_num, 1)
					
					probs = tf.nn.sigmoid(logits)  # (states_num * max_ac_num, 1)
					# operation: sample net2wider actions, (states_num, max_ac_num)
					update_ops["action_probs"] = tf.reshape(probs, shape=[states_num, -1])
			
			# net2wider train
			with tf.variable_scope("REINFORCE"):
				inputs["reward"] = tf.placeholder(tf.float32, [None], "reward")  # (states_num)
				inputs["action"] = tf.placeholder(tf.int32, [None, None], "action")  # (states_num, max_ac_num)
				inputs["action_mask"] = tf.placeholder(tf.float32, [None, None], "action_mask")
				inputs["episode_num"] = tf.placeholder(tf.float32, shape=(), name="episode_num")
				indices = tf.reshape(inputs["action"], [-1])  # (states_num * max_ac_num)
				indices = tf.stack([tf.range(array_ops.shape(indices)[0]), indices], 1)
				ex_probs = tf.concat([1 - probs, probs], axis=1)  # (states_num * max_ac_num, 2)
				action_probs = tf.gather_nd(ex_probs, indices)  # (states_num * max_ac_num)
				log_action_probs = tf.log(action_probs)  # (states_num * max_ac_num)
				log_action_probs = tf.reshape(log_action_probs,
											  shape=[states_num, -1])  # (states_num, max_ac_num)
				log_action_probs = tf.multiply(log_action_probs,
											   inputs["action_mask"])  # (states_num, max_ac_num)
				obj = tf.multiply(log_action_probs, tf.expand_dims(inputs["reward"], axis=1))
				obj = tf.reduce_sum(obj)
				if build_train:
					# overheads: _loss
					overheads["loss"] = -obj / inputs["episode_num"]
					# operation: _reinforce
					update_ops["reinforce"] = optimizer.minimize(overheads["loss"], self.overheads["global_step"])
			
			self.initializers.update(add_domain2dict(initializers, domain))
			self.inputs.update(add_domain2dict(inputs, domain))
			self.overheads.update(add_domain2dict(overheads, domain))
			self.update_ops.update(add_domain2dict(update_ops, domain))
			self.summaries.update(add_domain2dict(summaries, domain))
			
			return obj
	
	def net2deeper_decoder(self, states, domain=None, build_train=True):
		# states: (states_num, num_steps, units)
		initializers, inputs, overheads, update_ops, summaries = {}, {}, {}, {}, {}
		optimizer = TFUtils.build_optimizer(self.config["optimizer"])
		
		config = self.config["net2deeper"]
		output_initializer = self.get_initializer(config.get("output_initializer"), "output")
		with tf.variable_scope("Net2Deeper"):
			# sample action
			with tf.variable_scope("Sample"):
				seq_len = self.inputs["seq_len"][self.inputs["state_seg"]:]
				seq_num = array_ops.shape(seq_len)[0]
				gather_indices = tf.stack([tf.range(seq_num), seq_len - 1], 1)  # (states_num, 2)
				gather_output = tf.gather_nd(states, gather_indices)  # (states_num, units)
				
				cell_units = self.config["encoder"]["hidden_units"]
				fw_output = gather_output[:, 0:cell_units]
				bw_output = states[:, 0, :][:, cell_units:]
				gather_output = tf.concat([fw_output, bw_output], axis=1)  # (states_num, units)
				
				place_logits = tf.layers.dense(gather_output, config["place_out_dim"],
											   kernel_initializer=output_initializer["kernel"],
											   bias_initializer=output_initializer["bias"], name="place_predictor")
				place_probs = TFUtils.activation_func("softmax", place_logits)
				update_ops["place_probs"] = place_probs
				
				param_logits = tf.layers.dense(gather_output, config["param_out_dim"],
											   kernel_initializer=output_initializer["kernel"],
											   bias_initializer=output_initializer["bias"], name="param_predictor")
				param_probs = TFUtils.activation_func("softmax", param_logits)
				update_ops["param_probs"] = param_probs
			
			# net2deeper train
			with tf.variable_scope("REINFORCE"):
				inputs["reward"] = tf.placeholder(tf.float32, [None], "reward")  # (states_num)
				inputs["place_action"] = tf.placeholder(tf.int32, [None], "place_action")  # (states_num)
				inputs["param_action"] = tf.placeholder(tf.int32, [None], "param_action")  # (states_num)
				inputs["place_probs_mask"] = \
					tf.placeholder(tf.float32, [None, config["place_out_dim"]], "place_probs_mask")
				inputs["param_probs_mask"] = \
					tf.placeholder(tf.float32, [None, config["param_out_dim"]], "param_out_dim")
				inputs["place_loss_mask"] = tf.placeholder(tf.float32, [None], "place_loss_mask")  # (states_num)
				inputs["param_loss_mask"] = tf.placeholder(tf.float32, [None], "param_loss_mask")  # (states_num)
				inputs["episode_num"] = tf.placeholder(tf.float32, shape=(), name="episode_num")
				
				place_probs = tf.multiply(place_probs, inputs["place_probs_mask"])
				place_probs = tf.divide(place_probs, tf.reduce_sum(place_probs, axis=1, keep_dims=True))
				indices = tf.stack([tf.range(array_ops.shape(inputs["place_action"])[0]),
									inputs["place_action"]], axis=1)
				place_probs = tf.gather_nd(place_probs, indices)  # (states_num)
				log_place_probs = tf.log(place_probs)  # (states_num)
				log_place_probs = tf.multiply(log_place_probs, inputs["reward"])
				log_place_probs = tf.multiply(log_place_probs, inputs["place_loss_mask"])
				log_place_probs = tf.reduce_sum(log_place_probs)
				
				param_probs = tf.multiply(param_probs, inputs["param_probs_mask"])
				param_probs = tf.divide(param_probs, tf.reduce_sum(param_probs, axis=1, keep_dims=True))
				indices = tf.stack([tf.range(array_ops.shape(inputs["param_action"])[0]),
									inputs["param_action"]], axis=1)
				param_probs = tf.gather_nd(param_probs, indices)  # (states_num)
				log_param_probs = tf.log(param_probs)
				log_param_probs = tf.multiply(log_param_probs, inputs["reward"])
				log_param_probs = tf.multiply(log_param_probs, inputs["param_loss_mask"])
				log_param_probs = tf.reduce_sum(log_param_probs)
				
				obj = (log_place_probs + log_param_probs)
				if build_train:
					# overheads: _loss
					overheads["loss"] = -obj / inputs["episode_num"]
					# operation: _reinforce
					update_ops["reinforce"] = optimizer.minimize(overheads["loss"], self.overheads["global_step"])
		
		self.initializers.update(add_domain2dict(initializers, domain))
		self.inputs.update(add_domain2dict(inputs, domain))
		self.overheads.update(add_domain2dict(overheads, domain))
		self.update_ops.update(add_domain2dict(update_ops, domain))
		self.summaries.update(add_domain2dict(summaries, domain))
		
		return obj
	
	def net2wider_sample_action(self, net_seq, net_seq_len, valid_action, domain=None, _random=False):
		action_probs = self.sess.run(self.update_ops[domain_key("action_probs", domain)], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs["state_seg"]: len(net_seq),
			self.inputs[domain_key("valid_action", domain)]: valid_action
		})
		if _random:
			action = np.random.randint(0, 2, action_probs.shape)
		else:
			action = np.random.random_sample(action_probs.shape) <= action_probs
		return action.astype(np.int32)
	
	def net2wider_reinforce(self, net_seq, net_seq_len, action, action_mask, valid_action, reward, episode_num,
							domain=None):
		self.sess.run(self.update_ops[domain_key("reinforce", domain)], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs["state_seg"]: len(net_seq),
			self.inputs[domain_key("valid_action", domain)]: valid_action,
			self.inputs[domain_key("reward", domain)]: reward,
			self.inputs[domain_key("action", domain)]: action,
			self.inputs[domain_key("action_mask", domain)]: action_mask,
			self.inputs[domain_key("episode_num", domain)]: episode_num
		})
	
	def net2deeper_sample_action(self, net_seq, net_seq_len, domain=None, _random=False):
		place_probs, param_probs = self.sess.run([self.update_ops[domain_key("place_probs", domain)],
												  self.update_ops[domain_key("param_probs", domain)]], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs["state_seg"]: 0,
		})
		if _random:
			place_probs = np.ones_like(place_probs)
			param_probs = np.ones_like(param_probs)
		return place_probs, param_probs
	
	def net2deeper_reinforce(self, net_seq, net_seq_len, place_action, place_probs_mask, param_action, param_probs_mask,
							 reward, episode_num, place_loss_mask, param_loss_mask, domain=None):
		self.sess.run(self.update_ops[domain_key("reinforce", domain)], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs["state_seg"]: 0,
			self.inputs[domain_key("reward", domain)]: reward,
			self.inputs[domain_key("place_action", domain)]: place_action,
			self.inputs[domain_key("place_probs_mask", domain)]: place_probs_mask,
			self.inputs[domain_key("param_action", domain)]: param_action,
			self.inputs[domain_key("param_probs_mask", domain)]: param_probs_mask,
			self.inputs[domain_key("episode_num", domain)]: episode_num,
			self.inputs[domain_key("place_loss_mask", domain)]: place_loss_mask,
			self.inputs[domain_key("param_loss_mask", domain)]: param_loss_mask,
		})
	
	def net2net_reinforce(self, n2w_net_seq, n2w_net_seq_len, n2w_action, n2w_action_mask, n2w_valid_action, n2w_reward,
						  n2d_net_seq, n2d_net_seq_len, n2d_place_action, n2d_place_probs_mask, n2d_param_action,
						  n2d_param_probs_mask, n2d_reward, n2d_place_loss_mask, n2d_param_loss_mask, episode_num,
						  n2w_domain=None, n2d_domain=None, n2n_domain=None):
		state_seg = len(n2w_net_seq)
		net_seq = np.concatenate([n2w_net_seq, n2d_net_seq], axis=0)
		net_seq_len = np.concatenate([n2w_net_seq_len, n2d_net_seq_len], axis=0)
		self.sess.run(self.update_ops[domain_key("reinforce", n2n_domain)], feed_dict={
			self.inputs["input_seq"]: net_seq,
			self.inputs["seq_len"]: net_seq_len,
			self.inputs["state_seg"]: state_seg,
			self.inputs[domain_key("valid_action", n2w_domain)]: n2w_valid_action,
			self.inputs[domain_key("reward", n2w_domain)]: n2w_reward,
			self.inputs[domain_key("action", n2w_domain)]: n2w_action,
			self.inputs[domain_key("action_mask", n2w_domain)]: n2w_action_mask,
			self.inputs[domain_key("reward", n2d_domain)]: n2d_reward,
			self.inputs[domain_key("place_action", n2d_domain)]: n2d_place_action,
			self.inputs[domain_key("place_probs_mask", n2d_domain)]: n2d_place_probs_mask,
			self.inputs[domain_key("param_action", n2d_domain)]: n2d_param_action,
			self.inputs[domain_key("param_probs_mask", n2d_domain)]: n2d_param_probs_mask,
			self.inputs[domain_key("place_loss_mask", n2d_domain)]: n2d_place_loss_mask,
			self.inputs[domain_key("param_loss_mask", n2d_domain)]: n2d_param_loss_mask,
			self.inputs[domain_key("episode_num", n2n_domain)]: episode_num,
		})


class Net2NetAgent:
	def __init__(self, config):
		model_config = config["agent_model_config"]
		net_coder_config = model_config["net_coder"]
		self.net_coder = NetCoder(net_coder_config)
		model_config["input_dim"] = self.net_coder.net_seq_dim
		model_config["net2deeper"]["param_out_dim"] = self.net_coder.param_dim
		self.agent_model = Net2NetPolicyNet(model_config, config["model_restore_path"], config["exp_log_path"])
		
		if config["net_store_config"][0]:
			self.net_store_config = config["net_store_config"][1:]
			self.net_store = NetStore(config["net_store_config"][1])
		else:
			self.net_store = None
	
	def net2wider(self, net_configs, max_ac_num, store_states, store_actions, _random=False):
		num_steps = self.agent_model.config["encoder"]["num_steps"]
		batch_size = len(net_configs)
		
		net_seq, net_seq_len, valid_action = [0] * batch_size, [0] * batch_size, [0] * batch_size
		valid_layer, action_len = [0] * batch_size, [0] * batch_size
		for _j in range(batch_size):
			net_seq[_j], net_seq_len[_j] = self.net_coder.net_config2seq(net_configs[_j], num_steps)
			valid_action[_j], valid_layer[_j] = self.net_coder.widen_valid_layer(net_configs[_j])
			action_len[_j] = len(valid_action[_j])
		for _j in range(batch_size):
			valid_action[_j] += [0] * (max_ac_num - action_len[_j])
		net_seq, net_seq_len, valid_action, action_len = \
			np.asarray(net_seq), np.asarray(net_seq_len), np.asarray(valid_action), np.asarray(action_len)
		
		action = self.agent_model.net2wider_sample_action(net_seq, net_seq_len, valid_action, domain="net2wider",
														  _random=_random)
		action_mask = make_matrix_mask(batch_size, max_ac_num, upper=action_len)
		action *= action_mask
		
		store_states["net_seq"].append(net_seq)
		store_states["net_seq_len"].append(net_seq_len)
		store_actions["action"].append(action)
		store_actions["action_mask"].append(action_mask)
		store_actions["valid_action"].append(valid_action)
		
		net_configs = self.n2w_decode_net2wider_action(net_configs, valid_layer, action, action_len)
		return net_configs
	
	def n2w_decode_net2wider_action(self, net_configs, valid_layer, action, action_len):
		widen_nets = [0] * len(net_configs)
		for _i in range(len(net_configs)):
			widen_nets[_i] = net_configs[_i].copy()
			for _j in range(action_len[_i]):
				if action[_i][_j] == 1:
					layer_id = valid_layer[_i][_j]
					widen_nets[_i][layer_id] = self.net_coder.widen(net_configs[_i][layer_id])
		return widen_nets
	
	def n2w_update_agent(self, states, actions, reward):
		net_seq, net_seq_len = states["net_seq"], states["net_seq_len"]
		action, action_mask, valid_action = actions["action"], actions["action_mask"], actions["valid_action"]
		
		reward = np.stack([reward for _ in range(net_seq.shape[1])], axis=1).flatten()
		episode_num = 1  # len(net_seq)
		net_seq = np.reshape(net_seq, [-1] + list(net_seq.shape)[2:])
		net_seq_len = np.reshape(net_seq_len, [-1] + list(net_seq_len.shape)[2:])
		action = np.reshape(action, [-1] + list(action.shape)[2:])
		action_mask = np.reshape(action_mask, [-1] + list(action_mask.shape)[2:])
		valid_action = np.reshape(valid_action, [-1] + list(valid_action.shape)[2:])
		
		self.agent_model.net2wider_reinforce(net_seq, net_seq_len, action, action_mask, valid_action, reward,
											 episode_num, domain="net2wider")
	
	def net2deeper(self, net_configs, store_states, store_actions, _random=False):
		num_steps = self.agent_model.config["encoder"]["num_steps"]
		batch_size = len(net_configs)
		
		net_seq, net_seq_len = [0] * batch_size, [0] * batch_size
		for _j in range(batch_size):
			net_seq[_j], net_seq_len[_j] = self.net_coder.net_config2seq(net_configs[_j], num_steps)
		
		net_seq, net_seq_len = np.asarray(net_seq), np.asarray(net_seq_len)
		
		place_probs, param_probs = self.agent_model.net2deeper_sample_action(net_seq, net_seq_len, "net2deeper",
																			 _random=_random)
		
		place_action = {"action": [], "probs_mask": [], "loss_mask": []}
		param_action = {"action": [], "probs_mask": [], "loss_mask": []}
		deepen_nets = []
		for _i in range(batch_size):
			net_config = net_configs[_i]
			each_place_prob = place_probs[_i]
			each_param_prob = param_probs[_i]
			
			# decode net2deeper action
			valid_place = self.net_coder.deepen_valid_place(net_config)
			place_probs_mask = np.zeros([self.agent_model.config["net2deeper"]["place_out_dim"]])
			for place in valid_place:
				place_probs_mask[place] = 1
			place_valid_probs = np.asarray([each_place_prob[place] for place in valid_place])
			place_valid_probs /= np.sum(place_valid_probs)
			place_act = np.argmax(np.random.multinomial(1, place_valid_probs))
			place_act = valid_place[place_act]
			
			place_action["action"].append(place_act)
			place_action["probs_mask"].append(place_probs_mask)
			place_action["loss_mask"].append(1)
			
			new_layer = None
			for _j in range(place_act, -1, -1):
				if net_config[_j][0] == "C":
					param_vlist = self.net_coder.get_param_vlist("C", "KS")
					valid_code = np.arange(len(param_vlist))
					param_probs_mask = np.zeros([self.agent_model.config["net2deeper"]["param_out_dim"]])
					for code in valid_code:
						param_probs_mask[code] = 1
					param_valid_probs = np.asarray([each_param_prob[code] for code in valid_code])
					param_valid_probs /= np.sum(param_valid_probs)
					param_act = np.argmax(np.random.multinomial(1, param_valid_probs))
					param_act = valid_code[param_act]
					
					new_layer = ("C", {"KS": param_vlist[param_act], "FN": net_config[_j][1]["FN"]})
					param_action["action"].append(param_act)
					param_action["probs_mask"].append(param_probs_mask)
					param_action["loss_mask"].append(1)
					break
				elif net_config[_j][0] == "FC":
					param_probs_mask = np.ones([self.agent_model.config["net2deeper"]["param_out_dim"]])
					param_act = 0
					
					new_layer = ("FC", {"units": net_config[_j][1]["units"]})
					param_action["action"].append(param_act)
					param_action["probs_mask"].append(param_probs_mask)
					param_action["loss_mask"].append(0)
					break
			deepen_nets.append(net_config[:place_act + 1] + [new_layer] + net_config[place_act + 1:])
		
		store_states["net_seq"].append(net_seq)
		store_states["net_seq_len"].append(net_seq_len)
		
		store_actions["place_action"].append(np.asarray(place_action["action"]))
		store_actions["place_probs_mask"].append(np.asarray(place_action["probs_mask"]))
		
		store_actions["param_action"].append(np.asarray(param_action["action"]))
		store_actions["param_probs_mask"].append(np.asarray(param_action["probs_mask"]))
		
		store_actions["place_loss_mask"].append(np.asarray(place_action["loss_mask"]))
		store_actions["param_loss_mask"].append(np.asarray(param_action["loss_mask"]))
		return deepen_nets
	
	def n2d_update_agent(self, states, actions, reward):
		net_seq, net_seq_len = states["net_seq"], states["net_seq_len"]
		place_action, place_probs_mask = actions["place_action"], actions["place_probs_mask"]
		param_action, param_probs_mask = actions["param_action"], actions["param_probs_mask"]
		place_loss_mask, param_loss_mask = actions["place_loss_mask"], actions["param_loss_mask"]
		
		reward = np.stack([reward for _ in range(net_seq.shape[1])], axis=1).flatten()
		episode_num = 1  # len(net_seq)
		net_seq = np.reshape(net_seq, [-1] + list(net_seq.shape)[2:])
		net_seq_len = np.reshape(net_seq_len, [-1] + list(net_seq_len.shape)[2:])
		place_action = np.reshape(place_action, [-1] + list(place_action.shape)[2:])
		place_probs_mask = np.reshape(place_probs_mask, [-1] + list(place_probs_mask.shape)[2:])
		param_action = np.reshape(param_action, [-1] + list(param_action.shape)[2:])
		param_probs_mask = np.reshape(param_probs_mask, [-1] + list(param_probs_mask.shape)[2:])
		place_loss_mask = np.reshape(place_loss_mask, [-1] + list(place_loss_mask.shape)[2:])
		param_loss_mask = np.reshape(param_loss_mask, [-1] + list(param_loss_mask.shape)[2:])
		
		self.agent_model.net2deeper_reinforce(net_seq, net_seq_len, place_action, place_probs_mask, param_action,
											  param_probs_mask, reward, episode_num, place_loss_mask, param_loss_mask,
											  domain="net2deeper")
	
	def n2n_update_agent(self, n2w_states, n2w_actions, n2d_states, n2d_actions, reward):
		assert len(n2w_states["net_seq"]) == len(n2d_states["net_seq"])
		episode_num = 1  # len(n2w_states["net_seq"])
		
		n2w_net_seq, n2w_net_seq_len = n2w_states["net_seq"], n2w_states["net_seq_len"]
		n2w_action, n2w_action_mask, n2w_valid_action = n2w_actions["action"], n2w_actions["action_mask"], \
														n2w_actions["valid_action"]
		n2w_reward = np.stack([reward for _ in range(n2w_net_seq.shape[1])], axis=1).flatten()
		n2w_net_seq = np.reshape(n2w_net_seq, [-1] + list(n2w_net_seq.shape)[2:])
		n2w_net_seq_len = np.reshape(n2w_net_seq_len, [-1] + list(n2w_net_seq_len.shape)[2:])
		n2w_action = np.reshape(n2w_action, [-1] + list(n2w_action.shape)[2:])
		n2w_action_mask = np.reshape(n2w_action_mask, [-1] + list(n2w_action_mask.shape)[2:])
		n2w_valid_action = np.reshape(n2w_valid_action, [-1] + list(n2w_valid_action.shape)[2:])
		
		n2d_net_seq, n2d_net_seq_len = n2d_states["net_seq"], n2d_states["net_seq_len"]
		n2d_place_action, n2d_place_probs_mask = n2d_actions["place_action"], n2d_actions["place_probs_mask"]
		n2d_param_action, n2d_param_probs_mask = n2d_actions["param_action"], n2d_actions["param_probs_mask"]
		n2d_place_loss_mask, n2d_param_loss_mask = n2d_actions["place_loss_mask"], n2d_actions["param_loss_mask"]
		
		n2d_reward = np.stack([reward for _ in range(n2d_net_seq.shape[1])], axis=1).flatten()
		n2d_net_seq = np.reshape(n2d_net_seq, [-1] + list(n2d_net_seq.shape)[2:])
		n2d_net_seq_len = np.reshape(n2d_net_seq_len, [-1] + list(n2d_net_seq_len.shape)[2:])
		n2d_place_action = np.reshape(n2d_place_action, [-1] + list(n2d_place_action.shape)[2:])
		n2d_place_probs_mask = np.reshape(n2d_place_probs_mask, [-1] + list(n2d_place_probs_mask.shape)[2:])
		n2d_param_action = np.reshape(n2d_param_action, [-1] + list(n2d_param_action.shape)[2:])
		n2d_param_probs_mask = np.reshape(n2d_param_probs_mask, [-1] + list(n2d_param_probs_mask.shape)[2:])
		n2d_place_loss_mask = np.reshape(n2d_place_loss_mask, [-1] + list(n2d_place_loss_mask.shape)[2:])
		n2d_param_loss_mask = np.reshape(n2d_param_loss_mask, [-1] + list(n2d_param_loss_mask.shape)[2:])
		
		self.agent_model.net2net_reinforce(n2w_net_seq, n2w_net_seq_len, n2w_action, n2w_action_mask, n2w_valid_action,
										   n2w_reward, n2d_net_seq, n2d_net_seq_len, n2d_place_action,
										   n2d_place_probs_mask, n2d_param_action, n2d_param_probs_mask, n2d_reward,
										   n2d_place_loss_mask, n2d_param_loss_mask, episode_num,
										   n2w_domain="net2wider", n2d_domain="net2deeper", n2n_domain="net2net")
