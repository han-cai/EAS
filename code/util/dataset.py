import matplotlib.pyplot as plt
from util.data_processing import *


class ImageDataset:
	def __init__(self, dataset_dir=None, config=None, data=None):
		if data:
			self.data = data
		else:
			self.dataset_dir = dataset_dir
			# CIFAR-10: config = {"reader": cifar10, "scheme": ...}
			self.config = config
			# images: [image_num, row (height), column (width), channel (depth)]. label: [image_num, ]
			if self.config["reader"] == "cifar10":
				self.reader = Cifar10Reader(self.dataset_dir, self.config)
				self.data = {
					"train": self.reader.load_data("train"),
					"validate": self.reader.load_data("validate"),
					"test": self.reader.load_data("test"),
					"full-train": self.reader.load_data("full-train"),
					"label_name_map": self.reader.load_data("label_name_map"),
				}
			elif self.config["reader"] == "cifar100":
				self.reader = Cifar100Reader(self.dataset_dir, self.config)
				self.data = {
					"train": self.reader.load_data("train"),
					"validate": self.reader.load_data("validate"),
					"test": self.reader.load_data("test"),
					"full-train": self.reader.load_data("full-train"),
					"label_name_map": self.reader.load_data("label_name_map"),
				}
	
	def _line_generator_(self, subset_name, num_loops, per_loop_reload=False):
		for i in range(num_loops):
			if per_loop_reload:
				self.data[subset_name] = self.reader.load_data(subset_name)
			for x, y in zip(*self.data[subset_name]):
				yield x, y
	
	def _data_generator_(self, subset_name, batch_size, num_loops, per_loop_reload=False):
		line_gen = self._line_generator_(subset_name, num_loops, per_loop_reload)
		while True:
			input_images, correct_labels = [], []
			try:
				for i in range(batch_size):
					image, label = next(line_gen)
					input_images.append(image)
					correct_labels.append(label)
			except StopIteration:
				break
			yield np.asarray(input_images), np.asarray(correct_labels)
	
	def training_set(self, batch_size, num_loops=1, per_loop_reload=True, full=False):
		if full:
			return self._data_generator_("full-train", batch_size, num_loops, per_loop_reload)
		else:
			return self._data_generator_("train", batch_size, num_loops, per_loop_reload)
	
	def validation_set(self, batch_size):
		return self._data_generator_("validate", batch_size, 1)
	
	def test_set(self, batch_size):
		return self._data_generator_("test", batch_size, 1)
	
	def plot_image(self, image, label):
		image_shape = image.shape
		if image_shape[-1] == 1:
			plt.imshow(image[:, :, 0], cmap='binary')
		else:
			plt.imshow(image)
		plt.title("label: {}".format(self.data["label_name_map"][label]))
		plt.show()
	
	def batch_num_per_epoch(self, batch_size, subset_name="train"):
		batch_num_per_epoch = int(len(self.data[subset_name][0]) / batch_size)
		return batch_num_per_epoch


class SeqDataset:
	def __init__(self, dataset_dir=None, config=None, data=None):
		if data:
			self.data = data
		else:
			with open(os.path.join(dataset_dir, "train.seq"), "rb") as fin:
				train_dict = pickle.load(fin)
			with open(os.path.join(dataset_dir, "valid.seq"), "rb") as fin:
				valid_dict = pickle.load(fin)
			with open(os.path.join(dataset_dir, "test.seq"), "rb") as fin:
				test_dict = pickle.load(fin)
			self.dataset_dir = dataset_dir
			self.config = config
			self.data = {
				"train": (train_dict["data"], train_dict["seq_len"], train_dict["labels"]),
				"validate": (valid_dict["data"], valid_dict["seq_len"], valid_dict["labels"]),
				"test": (test_dict["data"], test_dict["seq_len"], test_dict["labels"])
			}  # data: [seq_num, max_len, n_input]. label: [seq_num, n_out]
	
	def _line_generator_(self, subset_name, num_loops, per_loop_reload=False):
		for i in range(num_loops):
			if per_loop_reload:
				self.data[subset_name] = shuffle(data_tuple=self.data[subset_name], shuffle_num=1)
			for seq, seq_len, y in zip(*self.data[subset_name]):
				yield seq, seq_len, y
	
	def _data_generator_(self, subset_name, batch_size, num_loops, per_loop_reload=False):
		line_gen = self._line_generator_(subset_name, num_loops, per_loop_reload)
		while True:
			input_seqs, seq_lens, correct_labels = [], [], []
			try:
				for i in range(batch_size):
					seq, seq_len, label = next(line_gen)
					input_seqs.append(seq)
					seq_lens.append(seq_len)
					correct_labels.append(label)
			except StopIteration:
				break
			yield np.asarray(input_seqs), np.asarray(seq_lens), np.asarray(correct_labels)
	
	def training_set(self, batch_size, num_loops=1, per_loop_reload=True):
		return self._data_generator_("train", batch_size, num_loops, per_loop_reload)
	
	def validation_set(self, batch_size):
		return self._data_generator_("validate", batch_size, 1)
	
	def test_set(self, batch_size):
		return self._data_generator_("test", batch_size, 1)
	
	def batch_num_per_epoch(self, batch_size, subset_name="train"):
		batch_num_per_epoch = int(len(self.data[subset_name][0]) / batch_size)
		return batch_num_per_epoch
