from os.path import realpath, dirname, join, exists
from os import makedirs
from subprocess import run

def _assert_existing(path):
	if not exists(path):
		makedirs(path)
	return path

_clear_files = ["training", "test", "validation", "checkpoint", "output", "snapshot"]

def _clear(path):
	for file in _clear_files:
		run(["rm", "-rf", join(path, file)])

class ExperimentDirectory:
	path = None

	def __init__(self, path, restore=False):
		self.path = realpath(path)
		if not exists(self.path):
			print("Path {} does not exist.".format(self.path))
			makedirs(self.path)
		# assert exists(self.path), "Path {} does not exist.".format(self.path)
		if not restore:
			_clear(self.path)

	@property
	def checkpoint(self):
		return CheckpointDirectory(join(self.path, "checkpoint"))

	@property
	def summary(self):
		return SummaryDirectory(self.path)

	@property
	def snapshot(self):
		return SnapshotDirectory(join(self.path, "snapshot"))

	@property
	def start_snapshot(self):
		return SnapshotDirectory(self.path)

	@property
	def output(self):
		return join(self.path, "output")

class SummaryDirectory:
	path = None

	def __init__(self, path):
		self.path = _assert_existing(realpath(path))

	@property
	def training(self):
		return join(self.path, "training")

	@property
	def validation(self):
		return join(self.path, "validation")

	@property
	def test(self):
		return join(self.path, "test")

class SnapshotDirectory:
	path = None

	def __init__(self, path):
		self.path = _assert_existing(realpath(path))

	@property
	def config(self):
		return join(self.path, "config")

	@property
	def init(self):
		return join(self.path, "init")

class CheckpointDirectory:
	path = None

	def __init__(self, path):
		self.path = _assert_existing(realpath(path))

	@property
	def ckpt(self):
		return join(self.path, "model.ckpt")
