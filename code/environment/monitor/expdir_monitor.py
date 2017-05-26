import json
from environment.monitor import RuntimeMonitor
from tf_network.expconfig import StdDividedConfig, RuntimeDividedConfig


class ExpdirMonitor:
	std_divided_config = None
	runtime_monitor = None
	expdir = None
	result = None
	
	def __init__(self, expdir, restore=False, dataset="cifar10"):
		self.std_divided_config = StdDividedConfig(expdir.start_snapshot)
		runtime_divided_config = RuntimeDividedConfig(self.std_divided_config)
		self.runtime_monitor = RuntimeMonitor(runtime_divided_config, expdir, restore, dataset)
		self.expdir = expdir
	
	def run(self):
		try:
			self.result = self.runtime_monitor.run()
			with open(self.expdir.output, "w") as f:
				json.dump(self.result, f, indent='\t')
			self.runtime_monitor.renew(self.std_divided_config)
			self.std_divided_config.dump(self.expdir.snapshot)
			return self.result
		except KeyboardInterrupt:
			raise KeyboardInterrupt
	
	def test(self):
		try:
			return self.runtime_monitor.test()
		except KeyboardInterrupt:
			raise KeyboardInterrupt
	
	def pure_train(self, valid=None):
		try:
			result = self.runtime_monitor.pure_train(valid)
			if result:
				with open(self.expdir.output, "w") as f:
					json.dump(result, f, indent='\t')
			self.runtime_monitor.renew(self.std_divided_config)
			self.std_divided_config.dump(self.expdir.snapshot)
			return result
		except KeyboardInterrupt:
			raise KeyboardInterrupt
