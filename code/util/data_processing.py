import numpy as np
import os
import pickle
import cv2
import sys
import urllib
import tarfile
import shutil

def shuffle(data_tuple, shuffle_num=1):
	perm = np.arange(len(data_tuple[0]))
	for _i in range(shuffle_num):
		np.random.shuffle(perm)
	return tuple(np.asarray(data)[perm] for data in data_tuple)


def random_flip_left_right(image):
	if np.random.random() < 0.5:
		image = cv2.flip(image, 1)
	return image


def random_flip_up_down(image):
	if np.random.random() < 0.5:
		image = cv2.flip(image, 0)
	return image


def random_crop(image, size):
	if len(image.shape) == 3:
		H, W, D = image.shape
		h, w, d = size
	else:
		H, W = image.shape
		h, w = size
	left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)
	return image[top:top + h, left:left + w]


def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
	return image[offset_height:offset_height + target_height, offset_width:offset_width + target_width]


def random_contrast(image, lower, upper):
	contrast_factor = np.random.uniform(lower, upper)
	avg = np.mean(image)
	return (image - avg) * contrast_factor + avg


def random_brightness(image, max_delta):
	delta = np.random.randint(-max_delta, max_delta)
	return image - delta


def random_blur(image, size):
	if np.random.random() < 0.5:
		image = cv2.blur(image, size)
	return image


def normalize(image):
	return image / 255.0


def per_image_whitening(image):
	return (image - np.mean(image)) / np.std(image)


def per_channel_normalization(image):
	if len(image.shape) != 3:
		raise ValueError
	else:
		whiten_image = np.zeros_like(image)
		for channel in range(image.shape[-1]):
			whiten_image[:, :, channel] = (image[:, :, channel] - np.mean(image[:, :, channel])) / np.std(image[:, :, channel])
	return whiten_image


class BasicReader:
	_shuffle_num = 1
	_valid_num = 5000
	
	SCHEME_DENSENET = 0
	SCHEME_TF = 1
	SCHEME_OTHER = 2
	SCHEME_NONE = 3
	
	def load_data(self, key):
		if key == "label_name_map":
			return self.cifar["label_name_map"]
		elif key == "test" or key == "validate":
			images, labels = self.cifar[key]
			images, labels = list(images), list(labels)
			for _i in range(len(images)):
				images[_i] = self.image_process(images[_i], is_training=False)
			images, labels = np.asarray(images), np.asarray(labels)
			return images, labels
		elif key == "train":
			images, labels = self.cifar[key]
			images, labels = list(images), list(labels)
			for _i in range(len(images)):
				images[_i] = self.image_process(images[_i], is_training=True)
			images, labels = shuffle(data_tuple=(images, labels), shuffle_num=self._shuffle_num)
			return images, labels
		elif key == "full-train":
			train_images, train_labels = self.cifar["train"]
			valid_images, valid_labels = self.cifar["validate"]
			images, labels = list(train_images) + list(valid_images), list(train_labels) + list(valid_labels)
			for _i in range(len(images)):
				images[_i] = self.image_process(images[_i], is_training=True)
			images, labels = shuffle(data_tuple=(images, labels), shuffle_num=self._shuffle_num)
			return images, labels
	
	def image_process(self, image, is_training=False):
		image = np.reshape(image, (3, 32, 32))  # [depth, height, width]
		image = np.transpose(image, (1, 2, 0))  # [height, width, depth]
		image = image.astype(np.float32)
		if self.config["scheme"] == self.SCHEME_DENSENET:
			# in DenseNet
			if is_training:
				I = np.zeros((40, 40, 3))
				I[4:36, 4:36, :] = image
				image = random_crop(I, (32, 32, 3))
				image = random_flip_left_right(image)
			image = per_channel_normalization(image)
		elif self.config["scheme"] == self.SCHEME_TF:
			# in TF tutorial
			if is_training:
				image = random_crop(image, (24, 24, 3))
				image = random_flip_left_right(image)
				image = random_brightness(image, max_delta=63)
				image = random_contrast(image, lower=0.2, upper=1.8)
			else:
				image = crop_to_bounding_box(image, 4, 4, 24, 24)
			image = per_image_whitening(image)
		elif self.config["scheme"] == self.SCHEME_OTHER:
			if is_training:
				image = random_crop(image, (24, 24, 3))
				image = random_flip_left_right(image)
			else:
				image = crop_to_bounding_box(image, 4, 4, 24, 24)
			image = per_image_whitening(image)
		return image


class Cifar10Reader(BasicReader):
	def __init__(self, dataset_dir, config):
		self.dataset_dir = dataset_dir
		self.config = config
		train_filenames = [os.path.join(self.dataset_dir, "data_batch_{}".format(i)) for i in range(1, 6)]
		test_filename = os.path.join(self.dataset_dir, "test_batch")
		label_name_file = os.path.join(self.dataset_dir, "batches.meta")
		for name in train_filenames + [test_filename, label_name_file]:
			if not os.path.isfile(name):
				self.download_and_extract()
				break
		# training & validation
		train_images, train_labels = [], []
		for name in train_filenames:
			with open(name, "rb") as fin:
				train_dict_batch = pickle.load(fin, encoding="latin")
				train_images.extend(train_dict_batch["data"])
				train_labels += train_dict_batch["labels"]
		train_images, train_labels = \
			shuffle(data_tuple=(train_images, train_labels), shuffle_num=self._shuffle_num)
		valid_images, valid_labels = train_images[:self._valid_num], train_labels[:self._valid_num]
		train_images, train_labels = train_images[self._valid_num:], train_labels[self._valid_num:]
		# test
		test_images, test_labels = [], []
		with open(test_filename, "rb") as fin:
			test_dict = pickle.load(fin, encoding="latin")
			test_images.extend(test_dict["data"])
			test_labels += test_dict["labels"]
		test_images, test_labels = np.asarray(test_images), np.asarray(test_labels)
		
		# label_name_map
		with open(label_name_file, "rb") as fin:
			meta_info = pickle.load(fin, encoding="latin")
			label_names = meta_info["label_names"]
			label_name_map = {i: label_names[i] for i in range(len(label_names))}
		
		cifar10 = {
			"train": (train_images, train_labels),
			"validate": (valid_images, valid_labels),
			"test": (test_images, test_labels),
			"label_name_map": label_name_map,
		}
		self.cifar = cifar10

	def download_and_extract(self):
		destination_dir = self.dataset_dir
		if not os.path.exists(destination_dir):
			os.makedirs(destination_dir)
		file_name = "cifar-10-python.tar.gz"
		
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (file_name,
															 float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
			
		DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
		filepath = os.path.join(destination_dir, file_name)
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', file_name, statinfo.st_size, 'bytes.')
		
		tarfile.open(filepath, "r:gz").extractall("../tmp/")
		
		tmp_dir = os.path.join("../tmp/", "cifar-10-batches-py")
		for file in os.listdir(tmp_dir):
			os.rename(os.path.join(tmp_dir, file), os.path.join(destination_dir, file))
		shutil.rmtree("../tmp/")
		os.remove(filepath)
		
		
class Cifar100Reader(BasicReader):
	def __init__(self, dataset_dir, config):
		self.dataset_dir = dataset_dir
		self.config = config
		train_filename = os.path.join(self.dataset_dir, "train")
		test_filename = os.path.join(self.dataset_dir, "test")
		label_name_file = os.path.join(self.dataset_dir, "meta")
		
		# training & validation
		train_images, train_labels = [], []
		with open(train_filename, "rb") as fin:
			train_dict_batch = pickle.load(fin, encoding="latin")
			train_images.extend(train_dict_batch["data"])
			train_labels += train_dict_batch["fine_labels"]
		
		train_images, train_labels = \
			shuffle(data_tuple=(train_images, train_labels), shuffle_num=self._shuffle_num)
		valid_images, valid_labels = train_images[:self._valid_num], train_labels[:self._valid_num]
		train_images, train_labels = train_images[self._valid_num:], train_labels[self._valid_num:]
		# test
		test_images, test_labels = [], []
		with open(test_filename, "rb") as fin:
			test_dict = pickle.load(fin, encoding="latin")
			test_images.extend(test_dict["data"])
			test_labels += test_dict["fine_labels"]
		test_images, test_labels = np.asarray(test_images), np.asarray(test_labels)
		
		# label_name_map
		with open(label_name_file, "rb") as fin:
			meta_info = pickle.load(fin, encoding="latin")
			label_names = meta_info["fine_label_names"]
			label_name_map = {i: label_names[i] for i in range(len(label_names))}
		
		cifar100 = {
			"train": (train_images, train_labels),
			"validate": (valid_images, valid_labels),
			"test": (test_images, test_labels),
			"label_name_map": label_name_map,
		}
		self.cifar = cifar100
