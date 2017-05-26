import tensorflow as tf
import tf_network.tf_utils as TFUtils
from tensorflow.python.framework.errors_impl import NotFoundError
import os


class RNNSeq:
	def __init__(self, config, restore_path=None, logdir=None):
		self.config = config
		self.graph = tf.Graph()
		self.initializers, self.inputs, self.overheads, self.update_ops, self.summaries = {}, {}, {}, {}, {}
		self.default_initializer = TFUtils.build_initializer(
			{"type": "random_uniform", "stddev": self.config["stddev"]})
		self.embed_scope, self.encoder_scope = "Embedding", "Encoder"
		with self.graph.as_default():
			self.overheads["global_step"] = TFUtils.get_global_step()  # global step
			self.inputs["is_training"] = tf.placeholder(tf.bool, shape=(), name="is_training")  # is_training
			self.build_graph()  # build the graph, need to be overridden, TODO
			self.initializers["global_initializer"] = tf.global_variables_initializer()  # global variable initializer
			self.saver = tf.train.Saver()  # model saver
		
		sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(graph=self.graph, config=sess_config)
		if restore_path:
			try:
				self.saver.restore(self.sess, restore_path)
				print("Model restored from file: {}.".format(restore_path))
			except:
				self.sess.run(self.initializers["global_initializer"])
				print("{} not exist. Model randomly initialized.".format(restore_path))
		else:
			self.sess.run(self.initializers["global_initializer"])
			print("Model randomly initialized.")
		if logdir:
			self.summary_writer = tf.summary.FileWriter(logdir, graph=self.graph)
			print("Logs in {}.".format(logdir))
	
	def build_graph(self):
		pass
	
	def get_initializer(self, initializer_config, domain=""):
		if initializer_config:
			try:
				if domain == "dense" or domain == "output":
					initializer = {"kernel": TFUtils.build_initializer(initializer_config["kernel"]),
								   "bias": TFUtils.build_initializer(initializer_config["bias"])}
				else:
					initializer = TFUtils.build_initializer(initializer_config)
			except:
				initializer = self.get_initializer(None, domain)
		elif domain == "dense" or domain == "output":
			initializer = {"kernel": self.default_initializer, "bias": tf.zeros_initializer()}
		else:
			initializer = self.default_initializer
		
		return initializer
	
	def build_encoder(self):
		num_steps = self.config["encoder"]["num_steps"]  # maximum sequence length
		
		self.inputs["seq_len"] = tf.placeholder(tf.int32, [None], "seq_len")  # (seq_num, )
		# Embedding
		if self.config.get("embedding") and self.config["embedding"]["use_embedding"]:
			self.inputs["input_seq"] = tf.placeholder(tf.int32, [None, num_steps], "input_seq")  # (seq_num, num_steps)
			embedding_config = self.config["embedding"]
			embedding_initializer = self.get_initializer(embedding_config.get("initializer"))
			encoder_inputs, self.config["input_dim"] = TFUtils.embedding_layer(
				self.inputs["input_seq"], embedding_config, embedding_initializer, reuse=False, scope=self.embed_scope)
		# (seq_num, num_steps, n_input = embedding_dim)
		else:
			self.inputs["input_seq"] = tf.placeholder(tf.float32, shape=[None, num_steps, self.config["input_dim"]],
													  name="input_seq")  # (seq_num, num_steps, n_input)
			encoder_inputs = self.inputs["input_seq"]
		
		# Encoder
		encoder_config = self.config["encoder"]
		encoder_initializer = self.get_initializer(encoder_config.get("initializer"))
		encoder_outputs, encoder_state, encoder_cell = \
			TFUtils.rnn_encoder(encoder_inputs, self.inputs["seq_len"], encoder_config, encoder_initializer,
								reuse=False, scope=self.encoder_scope)
		return encoder_outputs, encoder_state, encoder_cell
	
	def close(self):
		self.sess.close()
		self.summary_writer.close()
	
	def save_model(self, save_path, save_name="model", global_step=None):
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if global_step:
			self.saver.save(self.sess, save_path + save_name, global_step=global_step)
		else:
			self.saver.save(self.sess, save_path + save_name)

	def reset_weights(self):
		self.sess.run(self.initializers["global_initializer"])
