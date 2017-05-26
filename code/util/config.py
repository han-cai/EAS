import tensorflow as tf
import numpy as np
import os

# random seed
random_seed = 0
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# path configuration
dataset_path = {
	"cifar10":  "../Datasets/CIFAR/CIFAR-10/",
	"cifar100": "../Datasets/CIFAR/CIFAR-100/",
	"mnist": "../Datasets/MNIST/",
	"svhn-full": "../Datasets/SVHN/full/",
	"svhn-crop": "../Datasets/SVHN/crop/",
}


def multiple_replace(text, replace_list):
	for old, new in replace_list:
		text = text.replace(old, new)
	return text


def make_matrix_mask(height, width, upper=None, lower=None):
	if upper is None:
		upper = [width] * height
	if lower is None:
		lower = [0] * height
	mask = np.asarray([[(lower[j] <= i < upper[j]) and 1 or 0 for i in range(width)] for j in range(height)])
	return mask


def add_domain2dict(_dict, domain):
	if domain:
		domain_dict = {}
		for key in _dict:
			domain_dict["{}/{}".format(domain, key)] = _dict[key]
		return domain_dict
	else:
		return _dict


def domain_key(key, domain):
	if domain:
		return "{}/{}".format(domain, key)
	else:
		return key
