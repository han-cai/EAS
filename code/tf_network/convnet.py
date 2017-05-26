import tensorflow as tf
from tqdm import tqdm
from sys import stderr
from tf_network.expconfig import RenewNetworkConfig
from util.data_processing import Cifar10Reader
from tensorflow.python.ops import array_ops

tf.GraphKeys.ACCURACY = "accuracy"
tf.GraphKeys.GLOBAL_INITIALIZER = "global_initializer"
tf.GraphKeys.LOCAL_INITIALIZER = "local_initializer"


class InputLayer:
	inputs = None
	labels = None
	training = None
	
	def __init__(self, inputs, labels, training=None):
		self.inputs = inputs
		self.labels = labels
		self.training = training
	
	def feed_dict(self, inputs, labels, training=None):
		feed_dict = {
			self.inputs: inputs,
			self.labels: labels
		}
		if self.training is not None:
			feed_dict.update({self.training: training})
		return feed_dict


def image_process(images, scheme, is_training):
	scope_name = "Training" if is_training else "Test"
	with tf.variable_scope(scope_name):
		if scheme == Cifar10Reader.SCHEME_DENSENET:
			if is_training:
				images = tf.image.pad_to_bounding_box(images, 4, 4, 40, 40)
				images = tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), images)
				images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
			images_mean = tf.reduce_mean(images, axis=[1, 2], keep_dims=True)
			images_var = tf.reduce_mean(tf.square(images), axis=[1, 2], keep_dims=True) - tf.square(images_mean)
			images_var = tf.nn.relu(images_var)
			images_stddev = tf.sqrt(images_var)
			images = (images - images_mean) / images_stddev
	return images

class Network:
	graph = None
	input_layer = None
	
	def __init__(self, config, image_scheme):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.build_graph(config, image_scheme)
	
	def build_graph(self, config, image_scheme):
		with tf.variable_scope("Inputs"):
			inputs = tf.placeholder(tf.float32, shape=[None, *config.image_size, 3], name="inputs")
			labels = tf.placeholder(tf.int32, shape=[None], name="labels")
			training = tf.placeholder(tf.bool, shape=[], name="training")
			self.input_layer = InputLayer(inputs, labels, training)
		
		# inputs = tf.cond(training, lambda: image_process(inputs, image_scheme, True), lambda: image_process(inputs, image_scheme, False), name="ImageProcessing")

		with tf.variable_scope("Body"):
			scores = config.apply(inputs, training)
		
		global_step = tf.get_variable(name="global_step", dtype=tf.int32, shape=(), trainable=False,
									  initializer=tf.zeros_initializer(),
									  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
		
		with tf.variable_scope("Overheads"):
			one_hot_labels = tf.one_hot(labels, depth=scores.shape.as_list()[1], name="one_hot_labels")
			softmax_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=scores, name="softmax_loss"))
			reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			loss = softmax_loss if len(reg_losses) == 0 else softmax_loss + config.reg_type(reg_losses)
			predictions = tf.argmax(scores, axis=-1, name="predictions")
			
			config.minimize(loss, global_step=global_step)
		
		with tf.variable_scope("Evaluations") as scope:
			avg_loss, _ = tf.metrics.mean(loss, name="avg_loss", updates_collections=tf.GraphKeys.UPDATE_OPS)
			accuracy, _ = tf.metrics.accuracy(labels=labels, predictions=predictions, name="accuracy",
											  updates_collections=tf.GraphKeys.UPDATE_OPS)
			tf.add_to_collection(tf.GraphKeys.ACCURACY, accuracy)
		
		tf.add_to_collection(tf.GraphKeys.GLOBAL_INITIALIZER, tf.global_variables_initializer())
		tf.add_to_collection(tf.GraphKeys.LOCAL_INITIALIZER, tf.local_variables_initializer())
		
		with tf.variable_scope("Summaries"):
			tf.summary.scalar("avg_loss", avg_loss)
			tf.summary.scalar("accuracy", accuracy)
			tf.add_to_collection(tf.GraphKeys.SUMMARY_OP, tf.summary.merge_all())
		
		tf.add_to_collection(tf.GraphKeys.SAVERS, tf.train.Saver())


class GraphKeys:
	_graph = None
	
	def __init__(self, graph):
		self._graph = graph
	
	@property
	def saver(self):
		return self._graph.get_collection(tf.GraphKeys.SAVERS)[0]
	
	@property
	def global_step(self):
		return self._graph.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
	
	@property
	def global_initializer(self):
		return self._graph.get_collection(tf.GraphKeys.GLOBAL_INITIALIZER)[0]
	
	@property
	def local_initializer(self):
		return self._graph.get_collection(tf.GraphKeys.LOCAL_INITIALIZER)[0]
	
	@property
	def accuracy(self):
		return self._graph.get_collection(tf.GraphKeys.ACCURACY)[0]
	
	@property
	def summary_op(self):
		return self._graph.get_collection(tf.GraphKeys.SUMMARY_OP)[0]
	
	@property
	def update_ops(self):
		return self._graph.get_collection(tf.GraphKeys.UPDATE_OPS)
	
	@property
	def train_op(self):
		return self._graph.get_collection(tf.GraphKeys.TRAIN_OP)[0]


class SummaryManager:
	graph = None
	summary_op = None
	global_step = None
	summary_directory = None
	
	training = None
	validation = None
	test = None
	
	def __init__(self, graph, summary_op, global_step, summary_directory):
		self.graph = graph
		self.summary_op = summary_op
		self.global_step = global_step
		self.graph_keys = GraphKeys(graph)
		self.summary_directory = summary_directory
	
	def __enter__(self):
		self.training = tf.summary.FileWriter(self.summary_directory.training, graph=self.graph)
		self.validation = tf.summary.FileWriter(self.summary_directory.validation)
		self.test = tf.summary.FileWriter(self.summary_directory.test)
		return self
	
	def __exit__(self, *args):
		self.training.close()
		self.validation.close()
		self.test.close()
	
	def add(self, fw, sess):
		fw.add_summary(sess.run(self.summary_op), global_step=sess.run(self.global_step))


class SaverManager:
	saver = None
	checkpoint = None
	
	def __init__(self, saver, global_step, checkpoint):
		self.saver = saver
		self.checkpoint = checkpoint
	
	def save(self, sess):
		self.saver.save(sess, self.checkpoint.ckpt)
	
	def restore(self, sess):
		try:
			self.saver.restore(sess, self.checkpoint.ckpt)
			print("Restore from file {}.".format(self.checkpoint.ckpt), file=stderr)
		except:
			print("Checkpoint not found.", file=stderr)


class Session:
	expdir = None
	data = None
	config = None
	graph = None
	graph_keys = None
	saver_manager = None
	feed_dict = None
	
	def __init__(self, data, model, expdir):
		self.expdir = expdir
		self.data = data
		self.graph = model.graph
		self.graph_keys = GraphKeys(model.graph)
		self.saver_manager = SaverManager(self.graph_keys.saver, self.graph_keys.global_step, expdir.checkpoint)
		self.feed_dict = model.input_layer.feed_dict
	
	def create_session(self):
		config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		config.gpu_options.allow_growth = True
		return tf.Session(graph=self.graph, config=config)
	
	def evaluation(self, fw_manager, fw, data_gen):
		with self.create_session() as sess:
			sess.run([self.graph_keys.global_initializer, self.graph_keys.local_initializer])
			self.saver_manager.restore(sess)
			for inputs, labels in data_gen:
				sess.run(self.graph_keys.update_ops, self.feed_dict(inputs, labels, False))
			fw_manager.add(fw, sess)
			return float(sess.run(self.graph_keys.accuracy))
	
	def train(self, training_loop, validation_loop, restore):
		validation_acc_list = [0.0, ]
		
		with SummaryManager(self.graph, self.graph_keys.summary_op, self.graph_keys.global_step,
							self.expdir.summary) as fw_manager:
			with self.create_session() as training_sess:
				training_sess.run([self.graph_keys.global_initializer, self.graph_keys.local_initializer])
				if restore:
					self.saver_manager.restore(training_sess)
				for inputs, labels in tqdm(self.data.training()):
					if training_sess.run(self.graph_keys.global_step) % training_loop == 0:
						training_sess.run(self.graph_keys.local_initializer)
					training_sess.run([self.graph_keys.train_op, *self.graph_keys.update_ops],
									  self.feed_dict(inputs, labels, True))
					if training_sess.run(self.graph_keys.global_step) % training_loop == 0:
						fw_manager.add(fw_manager.training, training_sess)
					if training_sess.run(self.graph_keys.global_step) % validation_loop == 0:
						self.saver_manager.save(training_sess)
						validation_acc_list.append(
							self.evaluation(fw_manager, fw_manager.validation, self.data.validation()))
				fw_manager.add(fw_manager.training, training_sess)
				self.saver_manager.save(training_sess)
				validation_acc_list.append(self.evaluation(fw_manager, fw_manager.validation, self.data.validation()))
				test_acc = self.evaluation(fw_manager, fw_manager.test, self.data.test())
		
		return max(validation_acc_list), max(validation_acc_list[-1:]), test_acc
	
	def test(self, restore):
		with self.create_session() as test_sess:
			test_sess.run(self.graph_keys.global_initializer)
			if restore:
				self.saver_manager.restore(test_sess)
			acc = []
			for data_gen in [self.data.test()]:
				test_sess.run(self.graph_keys.local_initializer)
				for inputs, labels in data_gen:
					test_sess.run(self.graph_keys.update_ops, self.feed_dict(inputs, labels, False))
				acc.append(float(test_sess.run(self.graph_keys.accuracy)))
		return acc
	
	def pure_train(self, restore, valid=None):
		with self.create_session() as training_sess:
			training_sess.run([self.graph_keys.global_initializer, self.graph_keys.local_initializer])
			if restore:
				self.saver_manager.restore(training_sess)
			for inputs, labels in tqdm(self.data.training()):
				training_sess.run(self.graph_keys.train_op, self.feed_dict(inputs, labels, True))
			acc = None
			if valid == "valid":
				training_sess.run(self.graph_keys.local_initializer)
				data_gen = self.data.validation()
				for inputs, labels in data_gen:
					training_sess.run(self.graph_keys.update_ops, self.feed_dict(inputs, labels, False))
				acc = float(training_sess.run(self.graph_keys.accuracy))
			elif valid == "test":
				training_sess.run(self.graph_keys.local_initializer)
				data_gen = self.data.test()
				for inputs, labels in data_gen:
					training_sess.run(self.graph_keys.update_ops, self.feed_dict(inputs, labels, False))
				acc = float(training_sess.run(self.graph_keys.accuracy))
			self.saver_manager.save(training_sess)
		return acc
			
	def renew(self, runtime_divided_config, std_divided_config):
		with self.create_session() as sess:
			self.saver_manager.restore(sess)
			RenewNetworkConfig(runtime_divided_config.runtime_network_config, std_divided_config.network_config,
							   self.graph, "Body").apply(sess)
