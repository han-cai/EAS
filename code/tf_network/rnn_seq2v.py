from tf_network.rnn_seq import RNNSeq
import tf_network.tf_utils as TFUtils
import tensorflow as tf
from tensorflow.python.ops import array_ops
from util.config import add_domain2dict, domain_key
import numpy as np


class RNNSeq2V(RNNSeq):
	def build_graph(self):
		encoder_outputs, _, _ = self.build_encoder()
		if self.config.get("build_train") is None:
			build_train = True
		else:
			build_train = self.config.get("build_train")
		self.seq2v_decoder_(encoder_outputs, self.config["task"], domain=None, build_train=build_train)
	
	def seq2v_decoder_(self, encoder_outputs, task, domain=None, build_train=True):
		initializers, inputs, overheads, update_ops, summaries = {}, {}, {}, {}, {}
		decoder_inputs = tf.stack(encoder_outputs, axis=0)  # (num_steps, seq_num, hidden_units)
		decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2])  # (seq_num, num_steps, hidden_units)
		
		encoder_config = self.config["encoder"]
		seq2v_config = self.config["seq2v"]
		with tf.variable_scope(task):
			if seq2v_config["use_all_outputs"]:
				decoder_inputs = tf.reshape(decoder_inputs, [-1, np.prod(decoder_inputs.shape.as_list()[1:])])
			else:
				seq_num = array_ops.shape(self.inputs["seq_len"])[0]
				gather_indices = tf.stack([tf.range(seq_num), self.inputs["seq_len"] - 1], 1)  # (seq_num, 2)
				gather_output = tf.gather_nd(decoder_inputs, gather_indices)  # (seq_num, hidden_units)
				if encoder_config["rnn_type"] == "BiRNN":
					hidden_units = self.config["encoder"]["hidden_units"]
					fw_output = gather_output[:, 0:hidden_units]
					bw_output = decoder_inputs[:, 0, :][:, hidden_units:]
					decoder_inputs = tf.concat([fw_output, bw_output], axis=1)
				else:
					decoder_inputs = gather_output
			
			if seq2v_config.get("no_out_bias"):
				out_bias = False
			else:
				out_bias = True
			# dense and output layers: (seq_num, out_dim)
			dense_initializer = self.get_initializer(seq2v_config.get("dense_initializer"), "dense")
			output_initializer = self.get_initializer(seq2v_config.get("output_initializer"), "output")
			decoder_outputs = TFUtils.classifier(decoder_inputs, seq2v_config["output_dim"], seq2v_config["dense"],
												 dense_initializer, output_initializer,
												 reuse=False, out_bias=out_bias, scope="Output")
			
			if "regression" in task:
				inputs["labels"] = tf.placeholder(tf.float32, shape=[None, seq2v_config["output_dim"]], name="labels")
			else:
				inputs["labels"] = tf.placeholder(tf.int32, shape=[None], name="labels")
			
			with tf.variable_scope("Overheads"):
				if task == "regression":
					labels = inputs["labels"]
					square_loss = tf.square(decoder_outputs - labels, name="square_loss")
					loss = tf.reduce_mean(square_loss)
					predictions = decoder_outputs
				else:
					labels = tf.one_hot(inputs["labels"], depth=seq2v_config["output_dim"], name="one_hot_labels")
					softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=decoder_outputs,
																		   name="softmax_loss")
					loss = tf.reduce_mean(softmax_loss)
					predictions = tf.argmax(decoder_outputs, axis=-1, name="predictions")
				overheads["loss"], overheads["predictions"] = loss, predictions
				if build_train:
					optimizer = TFUtils.build_optimizer(self.config["optimizer"])
					update_ops["minimize"] = optimizer.minimize(loss, global_step=self.overheads["global_step"])
			if build_train:
				TFUtils.train_loop(update_ops["minimize"], task, initializers, inputs, overheads, update_ops, summaries)
			TFUtils.test_loop(task, initializers, inputs, overheads, update_ops, summaries, stage="test")
			TFUtils.test_loop(task, initializers, inputs, overheads, update_ops, summaries, stage="validate")
		
		self.initializers.update(add_domain2dict(initializers, domain))
		self.inputs.update(add_domain2dict(inputs, domain))
		self.overheads.update(add_domain2dict(overheads, domain))
		self.update_ops.update(add_domain2dict(update_ops, domain))
		self.summaries.update(add_domain2dict(summaries, domain))
	
	def seq2v_train(self, train_data_gen, valid_data_maker, statistics_loop, domain=None, save_config=None):
		while self._seq2v_train_loop_(train_data_gen, valid_data_maker, statistics_loop, domain, save_config):
			pass
	
	def _seq2v_train_loop_(self, train_data_gen, valid_data_maker, statistics_loop, domain=None, save_config=None):
		RUNNING_FLAG = False
		self.sess.run(self.initializers[domain_key("train", domain)])
		for i, (input_seq, seq_len, labels) in zip(range(statistics_loop), train_data_gen):
			RUNNING_FLAG = True
			self.sess.run(self.update_ops[domain_key("train", domain)], feed_dict={
				self.inputs["input_seq"]: input_seq,
				self.inputs["is_training"]: True,
				self.inputs["seq_len"]: seq_len,
				self.inputs[domain_key("labels", domain)]: labels
			})
		if RUNNING_FLAG:
			global_step = self.sess.run(self.overheads["global_step"])
			self.summary_writer.add_summary(self.sess.run(self.summaries[domain_key("train", domain)]), global_step)
			
			self.sess.run(self.initializers[domain_key("validate", domain)])
			for input_seq, seq_len, labels in valid_data_maker():
				self.sess.run(self.update_ops[domain_key("validate", domain)], feed_dict={
					self.inputs["input_seq"]: input_seq,
					self.inputs["is_training"]: False,
					self.inputs["seq_len"]: seq_len,
					self.inputs[domain_key("labels", domain)]: labels
				})
			print("validation metric: {}.".format(self.sess.run(self.overheads[domain_key("validate_metric", domain)])))
			self.summary_writer.add_summary(self.sess.run(self.summaries[domain_key("validate", domain)]), global_step)
			if save_config:
				save_model_path, save_step_size = save_config
				if global_step % save_step_size == 0:
					self.save_model(save_model_path)
		return RUNNING_FLAG
	
	def seq2v_test(self, test_data_generator, domain=None):
		self.sess.run(self.initializers[domain_key("test", domain)])
		for input_seq, seq_len, labels in test_data_generator:
			self.sess.run(self.update_ops[domain_key("test", domain)], feed_dict={
				self.inputs["input_seq"]: input_seq,
				self.inputs["is_training"]: False,
				self.inputs["seq_len"]: seq_len,
				self.inputs[domain_key("labels", domain)]: labels
			})
		print("test metric: {}.".format(self.sess.run(self.overheads[domain_key("test_metric", domain)])))
		self.summary_writer.add_summary(self.sess.run(self.summaries[domain_key("test", domain)]))
	
	def seq2v_query(self, input_seq, seq_len, domain=None):
		pVals = self.sess.run(self.overheads[domain_key("predictions", domain)], feed_dict={
			self.inputs["input_seq"]: input_seq,
			self.inputs["is_training"]: False,
			self.inputs["seq_len"]: seq_len
		})
		return pVals
