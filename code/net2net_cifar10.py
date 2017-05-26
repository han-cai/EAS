from main import run, ExpdirMonitor
from util.exdir import ExperimentDirectory
from tf_network.expconfig import WiderNetworkConfig, DeeperNetworkConfig, StdDividedConfig
from tqdm import tqdm
import json
from agent.net2net_agent import Net2NetAgent
from tf_network.expconfig.std_config import *
import numpy as np
from environment.virtual_env import VirtualEnv
from os.path import join, exists
from os import makedirs
import distributed


def std_config2netconfig(std_config):
	net_config = []
	for layer in std_config.network_config.layers[:-1]:
		if isinstance(layer, StdConvLayer):
			layer_config = ("C", {"KS": layer.kernel_size, "FN": layer.filters})
		elif isinstance(layer, StdDenseLayer):
			layer_config = ("FC", {"units": layer.units})
		elif isinstance(layer, StdPoolLayer):
			layer_config = ("P", {"KS": layer.pool_size, "ST": layer.strides})
		else:
			continue
		net_config.append(layer_config)
	return net_config

start_dir = "placeholder"
first_expdir = ExperimentDirectory(start_dir, restore=True)

second_dir = "placeholder"
if not exists(second_dir):
	makedirs(second_dir)
	val_log_file = open(join(second_dir, ".log"), "w")
	net_log_file = open(join(second_dir, "nets.log"), "w")
else:
	val_log_file = open(join(second_dir, ".log"), "a")
	net_log_file = open(join(second_dir, "nets.log"), "a")
	
_random = False

episodes = 20
wider_action_num = 3
deeper_action_num = 2
batch_size = 10
max_ac_num = 30

# agent
net_store_config = (True, join(second_dir, "net.store"))
agent_config = {
	"agent_model_config": "../model-config/net2net_agent_real_cifar10.json",
	"model_restore_path": join(second_dir, "agent/model"),
	"net_store_config": net_store_config,
	"exp_log_path": None,
	"exp_model_save_path": join(second_dir, "agent/"),
	"config_save_path": None
}
agent_model_config = json.load(open(agent_config["agent_model_config"], "r"))
if wider_action_num == 0:
	agent_model_config["mode"] = "net2deeper"
elif deeper_action_num == 0:
	agent_model_config["mode"] = "net2wider"
else:
	agent_model_config["mode"] = "net2net"
agent_config["agent_model_config"] = agent_model_config
agent = Net2NetAgent(agent_config)

best_val = 0
ema, alpha = 0, 0.5
for _i in tqdm(range(episodes)):
	print("episode {} start.".format(_i))
	std_divided_config = StdDividedConfig(first_expdir.start_snapshot)
	final_train_dict = {
		"batch_size": 64,
		"epochs": 20,
		"train_full": False,
		"training_loop": 70,
		"validation_loop": 3500,
		"scheme": std_divided_config.std_config.scheme,
		"image_size": std_divided_config.std_config.image_size,
		"minimize": [
			"momentum",
			0.001,
			{"momentum": 0.9, "use_nesterov": True},
			["piecewise", {"boundaries": [10500], "values": [0.025, 0.005]}]
		]
	}
	
	n2d_states = {"net_seq": [], "net_seq_len": []}
	n2d_actions = {"place_action": [], "place_probs_mask": [], "param_action": [], "param_probs_mask": [],
				   "place_loss_mask": [], "param_loss_mask": []}
	
	n2w_states = {"net_seq": [], "net_seq_len": []}
	n2w_actions = {"action": [], "action_mask": [], "valid_action": []}
	
	net_config = std_config2netconfig(std_divided_config)
	
	net_configs = [[(param, vlist.copy()) for param, vlist in net_config] for _ in range(batch_size)]
	std_divided_configs = [StdDividedConfig(first_expdir.start_snapshot) for _ in range(batch_size)]
	for _ in range(deeper_action_num):
		net_configs = agent.net2deeper(net_configs, n2d_states, n2d_actions, _random=_random)
		for _j in range(batch_size):
			deeper_idx = n2d_actions["place_action"][_][_j]
			new_layer = net_configs[_j][deeper_idx + 1]
			if new_layer[0] == "FC":
				deeper_idx += 1
				param_dict = {}
			else:
				param_dict = {"kernel_size": new_layer[1]["KS"]}
			DeeperNetworkConfig(std_divided_configs[_j].network_config).apply(deeper_idx, param_dict, noise=0)
	for _ in range(wider_action_num):
		net_configs = agent.net2wider(net_configs, max_ac_num, n2w_states, n2w_actions, _random=_random)
		for _j in range(batch_size):
			wider_list = n2w_actions["action"][_][_j, :len(net_configs[_j])]
			WiderNetworkConfig(std_divided_configs[_j].network_config).apply(wider_list, noise=1, drop=1)
		
	if deeper_action_num > 0:
		n2d_states["net_seq"] = np.stack(n2d_states["net_seq"], axis=1)
		n2d_states["net_seq_len"] = np.stack(n2d_states["net_seq_len"], axis=1)
		n2d_actions["place_action"] = np.stack(n2d_actions["place_action"], axis=1)
		n2d_actions["place_probs_mask"] = np.stack(n2d_actions["place_probs_mask"], axis=1)
		n2d_actions["param_action"] = np.stack(n2d_actions["param_action"], axis=1)
		n2d_actions["param_probs_mask"] = np.stack(n2d_actions["param_probs_mask"], axis=1)
		n2d_actions["place_loss_mask"] = np.stack(n2d_actions["place_loss_mask"], axis=1)
		n2d_actions["param_loss_mask"] = np.stack(n2d_actions["param_loss_mask"], axis=1)
	if wider_action_num > 0:
		n2w_states["net_seq"] = np.stack(n2w_states["net_seq"], axis=1)
		n2w_states["net_seq_len"] = np.stack(n2w_states["net_seq_len"], axis=1)
		n2w_actions["action"] = np.stack(n2w_actions["action"], axis=1)
		n2w_actions["action_mask"] = np.stack(n2w_actions["action_mask"], axis=1)
		n2w_actions["valid_action"] = np.stack(n2w_actions["valid_action"], axis=1)
	
	net_strs = []
	rewards = [0] * batch_size
	to_run = {}
	for _j in range(batch_size):
		net_config = net_configs[_j]
		net_str = agent.net_coder.net_config2str(net_config)
		net_val = agent.net_store.get_net_value(net_str)
		net_strs.append(net_str)
		if net_val is None:
			pure_store_dir = join(second_dir, net_str)
			store_output = join(pure_store_dir, ".output")
			if exists(store_output):
				with open(store_output, "r") as fin:
					line = fin.readline()
					net_val = float(line[:len(line) - 1])
					agent.net_store.add_net_value(net_str, net_val)
					rewards[_j] = net_val
			else:
				if net_str in to_run:
					to_run[net_str].append(_j)
				else:
					to_run[net_str] = [_j]
		else:
			net_val = np.mean(net_val)
			rewards[_j] = net_val

	task_list = []

	for net_str in to_run:
		idx = to_run[net_str]
		net_config = net_configs[idx[0]]
		
		std_divided_config = std_divided_configs[idx[0]]
		std_divided_config.std_config.__dict__.update(final_train_dict)

		pure_store_dir = join(second_dir, net_str)
		store_dir = ExperimentDirectory(pure_store_dir)
		std_divided_config.dump(store_dir.start_snapshot)
		
		task_list.append([pure_store_dir, net_str, idx])

	distributed.run(task_list)

	for _, net_str, idx, net_val in task_list:
		agent.net_store.add_net_value(net_str, net_val)
		for id in idx:
			rewards[id] = net_val

	rewards = np.asarray(rewards)
	best_val = max(best_val, np.max(rewards))
	print("{}: mean {}, max {}; \tnet store: {} {}".format(_i, np.mean(rewards), np.max(rewards),
														   *agent.net_store.statistics()))
	val_log_file.write(
		"{}\t{}\t{}: {}\n".format(_i, np.mean(rewards), best_val, "\t".join([str(reward) for reward in rewards])))
	net_log_file.write("{}:\t{}\n".format(_i, "\t".join(net_strs)))
	
	if ema == 0:
		ema = np.mean(rewards)
	else:
		ema += alpha * (np.mean(rewards) - ema)
	rewards -= ema
	if not _random:
		if wider_action_num == 0:
			agent.n2d_update_agent(n2d_states, n2d_actions, rewards)
		elif deeper_action_num == 0:
			agent.n2w_update_agent(n2w_states, n2w_actions, rewards)
		else:
			agent.n2n_update_agent(n2w_states, n2w_actions, n2d_states, n2d_actions, rewards)
	
	if (_i + 1) % 1 == 0:
		agent.net_store.save(second_dir)
		agent.agent_model.save_model(agent_config["exp_model_save_path"])
	val_log_file.flush()
	net_log_file.flush()
val_log_file.close()
net_log_file.close()
