import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json, argparse
import numpy as np, tensorflow as tf
from util.config import *
from environment.monitor import ExpdirMonitor
from util.exdir import ExperimentDirectory

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("expdir", type=str)
	parser.add_argument("--restore", action="store_true")
	parser.add_argument("--test", action="store_true")
	parser.add_argument("dataset", type=str, default="cifar10")
	return parser.parse_args()

def run(expdir, restore=False, mode="train", dataset="cifar10"):
	expdir_monitor = ExpdirMonitor(expdir, restore, dataset)
	if mode == "train":
		expdir_monitor.run()
		return expdir_monitor
	elif mode == "test":
		result = expdir_monitor.test()
		print("test acc: {}".format(*result))
		return result
	elif mode == "pure_train":
		expdir_monitor.pure_train()

def main():
	args = get_args()
	expdir = ExperimentDirectory(args.expdir, args.restore)
	run(expdir, args.restore, "test" if args.test else "train", dataset=args.dataset)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
