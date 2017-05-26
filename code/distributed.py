from subprocess import Popen, PIPE
from threading import Thread, Lock
from queue import Queue
from time import sleep
from sys import stderr
import random, os, re, json, shlex

class GpuChecker:
	def __init__(self, nvidia_getter, gpuid):
		self.nvidia_getter = nvidia_getter
		self.gpuid = gpuid

	def state_parser(self, state_str):
		result = []
		for line in state_str.split('\n'):
			pattern = re.search('.*?(\d*)C.*\|(.*?)MiB.*?/(.*?)MiB.*?\|.*?(\d*)\%', line)
			if pattern is not None:
				result.append([int(x) for x in pattern.groups()])
		assert self.gpuid < len(result), "Parsing error or not enough gpus."
		return result[self.gpuid]

	def instance_available(self, state_str):
		parse_result = self.state_parser(state_str)
		_, used_mem, total_mem, occupation = parse_result
		return used_mem / total_mem < 0.5 and occupation < 50

	def check(self):
		try:
			assert self.instance_available(self.nvidia_getter())
			sleep(0.5)
			assert self.instance_available(self.nvidia_getter())
			sleep(0.5)
			assert self.instance_available(self.nvidia_getter())
		except AssertionError:
			return False
		return True

class RemoteController:
	def __init__(self, remote, gpuid, executive):
		self.remote = remote
		self.gpuid = gpuid
		self.executive = executive

		self.gpu_checker = GpuChecker(lambda: self.run("nvidia-smi"), self.gpuid)

		self._lock = Lock()
		self._occupied = False

	@property
	def occupied(self):
		with self._lock:
			return self._occupied

	@occupied.setter
	def occupied(self, val):
		assert isinstance(val, bool), "Occupied must be True or False, but {} received.".format(val)
		with self._lock:
			self._occupied = val

	def run(self, cmd, stdin=None):
		proc = Popen('ssh {} {}'.format(self.remote, shlex.quote(cmd)), shell=True, stdin=PIPE, stdout=PIPE, universal_newlines=True)
		return proc.communicate(input=stdin)[0]

	@property
	def gpu_state(self):
		return self.gpu_checker.check()

	@property
	def exe_cmd(self):
		return "CUDA_VISIBLE_DEVICES={gpuid} python3 {executive}".format(
				executive=self.executive,
				gpuid=self.gpuid
			)

	def remote_executer(self, idx, expdir, queue):
		self.occupied = True
		cmd = self.exe_cmd
		print("{}: {} {}".format(self.remote, cmd, expdir), file=stderr)
		result = self.run(cmd, stdin=expdir)
		try:
			result = float(result)
			queue.put([idx, result])
			print("{}th task: {} is successfully executed, result is {}.".format(idx, expdir, result), file=stderr)
		except:
			queue.put([idx, expdir])
			print("{}th task: {} fails.".format(idx, expdir), file=stderr)
		self.occupied = False

	def execute(self, idx, expdir, queue):
		if self.occupied or not self.gpu_state:
			queue.put([idx, expdir])
		else:
			thr = Thread(target=self.remote_executer, args=(idx, expdir, queue))
			thr.start()

class ClusterController:
	def __init__(self, config_list):
		self.cluster = [RemoteController(*config) for config in config_list]

	def choice(self):
		return random.choice(self.cluster)

	def execute(self, idx, expdir, queue):
		self.choice().execute(idx, expdir, queue)

def run_tasks(config_list, expdir_list):
	controller = ClusterController(config_list)
	result_list = [None for _ in expdir_list]

	queue = Queue()
	for idx, expdir in enumerate(expdir_list):
		queue.put([idx, expdir])

	remained = len(result_list)
	while remained > 0:
		idx, val = queue.get()
		if isinstance(val, str):
			# expdir, need to execute
			controller.execute(idx, val, queue)
		elif isinstance(val, float):
			# result, need to be put in result_list
			result_list[idx] = val
			remained -= 1

	return result_list

config_file = os.path.join(os.path.expanduser("~"), "server_config")
with open(config_file, "r") as f:
	config_list = json.load(f)

def run(task_list):
	expdir_list = [expdir for expdir, *_ in task_list]
	result_list = run_tasks(config_list, expdir_list)
	for idx, _ in enumerate(task_list):
		task_list[idx].append(result_list[idx])
