from tf_network.convnet import Network, Session
from util.dataset import ImageDataset
from util.config import dataset_path


class Data:
	def __init__(self, data_config, dataset="cifar10"):
		self.data_config = data_config
		_dataset_dir = dataset_path[dataset]
		self.image_data = ImageDataset(dataset_dir=_dataset_dir,
									   config={"reader": dataset, "scheme": data_config.scheme})
	
	def training(self):
		return self.image_data.training_set(self.data_config.batch_size, self.data_config.epochs,
											full=self.data_config.train_full)
	
	def validation(self):
		return self.image_data.validation_set(self.data_config.batch_size)
	
	def test(self):
		return self.image_data.test_set(self.data_config.batch_size)


class RuntimeMonitor:
	runtime_divided_config = None
	restore = None
	sess = None
	result = None
	
	def __init__(self, runtime_divided_config, expdir, restore=False, dataset="cifar10"):
		self.runtime_divided_config = runtime_divided_config
		self.restore = restore
		
		data = Data(runtime_divided_config.runtime_data_config, dataset)
		model = Network(runtime_divided_config.runtime_network_config, data.data_config.scheme)
		self.sess = Session(data, model, expdir)
	
	def run(self):
		runtime_monitor_config = self.runtime_divided_config.runtime_monitor_config
		training_loop = runtime_monitor_config.training_loop
		validation_loop = runtime_monitor_config.validation_loop
		try:
			self.result = self.sess.train(training_loop, validation_loop, self.restore)
			return self.result
		except KeyboardInterrupt:
			raise KeyboardInterrupt
	
	def renew(self, std_divided_config):
		try:
			self.sess.renew(self.runtime_divided_config, std_divided_config)
		except KeyboardInterrupt:
			raise KeyboardInterrupt
	
	def test(self):
		try:
			return self.sess.test(self.restore)
		except KeyboardInterrupt:
			raise KeyboardInterrupt
	
	def pure_train(self, valid=None):
		try:
			result = self.sess.pure_train(self.restore, valid)
			return result
		except KeyboardInterrupt:
			raise KeyboardInterrupt
