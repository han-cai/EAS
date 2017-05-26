import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environment.monitor import ExpdirMonitor
from util.exdir import ExperimentDirectory

def run(expdir):
	expdir_monitor = ExpdirMonitor(expdir)
	net_val = expdir_monitor.pure_train(valid="valid")
	print(net_val)

def main():
	expdir = ExperimentDirectory(input().strip('\n'))
	run(expdir)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
