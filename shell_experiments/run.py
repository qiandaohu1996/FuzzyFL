# Translating the primary functions and variables from the previous shell script to Python

# Global variables
import os
import shutil


DATA = []
algos = []
algo = ""
lr = 0.1
bz = 256
local_steps = 1
n_rounds = 200
log_freq = 2
sampling_rates = [0.5]
pre_rounds_list = [50]
fuzzy_m_schedulers = ["constant"]
fuzzy_m_list = [1.8]
min_m_list = [1.5]
min_m = 1.5
trans_list = [0.75]
fuzzy_m_momentums = [0.8]
measurements = ["loss"]
mus = [0.1, 0.5]
comm_probs = [0.2, 0.5]
alphas = []
commands = []

# Function to set inner directory
def set_inner_dir(param,parameters):
	param_to_template = {
		"pre_rounds": "pre_%s",
		"fuzzy_m": "_m_%s",
		"min_fuzzy_m": "_minm_%s",
		"sampling_rate": "_samp_%s",
		"locally_tune_clients": "_adapt",
		"adaptive": "_adapt",
		"fuzzy_m_scheduler": "_sch_%s",
		"measurement": "_msu_%s",
		"fuzzy_m_momentum": "_mt_%s",
		"comm_prob": "comm_%s_",
		"n_clusters": "_cluster_%s",
		"mu": "mu_%s"
	}

	inner_dir = ""
	if param in parameters:
		if parameters[param] != "true":
			template = param_to_template.get(param, "_${param}_%s")  # Default template
			inner_dir += template % parameters[param]
		else:
			inner_dir += f"_{param}"
	return inner_dir
# Function to run a process based on the parsed command
def run(dataset, algo, sampling_rate, pre_rounds=None, fuzzy_m=None, trans=None, min_fuzzy_m=None,
        fuzzy_m_scheduler=None, fuzzy_m_momentum=None, measurement=None, alpha=None, adaptive=False, minibatch=False):
    # Creating directory path based on parameters
    inner_dir = ""
    if pre_rounds:
        inner_dir += f"_pre_{pre_rounds}"
    if fuzzy_m:
        inner_dir += f"_m_{fuzzy_m}"
    if trans:
        inner_dir += f"_trans_{trans}"
    if min_fuzzy_m:
        inner_dir += f"_minm_{min_fuzzy_m}"
    if fuzzy_m_scheduler:
        inner_dir += f"_sch_{fuzzy_m_scheduler}"
    if fuzzy_m_momentum:
        inner_dir += f"_mt_{fuzzy_m_momentum}"
    if measurement:
        inner_dir += f"_msu_{measurement}"
    if alpha:
        inner_dir += f"_alpha_{alpha}"
    if adaptive:
        inner_dir += "_adaptive"
    if minibatch:
        inner_dir += "_minibatch"
    
    dir_path = f"{dataset}/{algo}/gd/{inner_dir}"
    handle_directory(dir_path)

# Example usage
# run("dataset_name", "FuzzyFL", 0.5, pre_rounds="50", fuzzy_m="1.8", minibatch=True)


# Example usage
# run("dataset_name", "algorithm_name", "extra_arg1=value1", "extra_arg2=value2")

# Function to set inner directory using Python string formatting
def set_inner_dir(param, parameters):
	param_to_template = {
		"pre_rounds": "pre_{}",
		"fuzzy_m": "_m_{}",
		"min_fuzzy_m": "_minm_{}",
		"sampling_rate": "_samp_{}",
		"locally_tune_clients": "_adapt",
		"adaptive": "_adapt",
		"fuzzy_m_scheduler": "_sch_{}",
		"measurement": "_msu_{}",
		"fuzzy_m_momentum": "_mt_{}",
		"comm_prob": "comm_{}_",
		"n_clusters": "_cluster_{}",
		"mu": "mu_{}"
	}

	inner_dir = ""
	value = parameters.get(param)
	if value:
		template = param_to_template.get(param, "_{}_{}")  # Default template
		inner_dir += template.format(param, value) if value != "true" else f"_{param}"

	return inner_dir

# Function to handle directory operations (creation, deletion, renaming)
def handle_directory(dir_path):
	# Confirming directory deletion
	if os.path.exists(dir_path):
		user_input = input(f"Directory {dir_path} exists. Remove it? (y/N): ")
		if user_input.lower() == 'y':
			shutil.rmtree(dir_path)
			print(f"Directory {dir_path} removed.")
		elif user_input.lower() == 'n':
			suffix = 1
			new_dir_path = f"{dir_path}_{suffix}"
			while os.path.exists(new_dir_path):
				suffix += 1
				new_dir_path = f"{dir_path}_{suffix}"
			os.rename(dir_path, new_dir_path)
			print(f"Directory {dir_path} renamed to {new_dir_path}")

	os.makedirs(dir_path)
	print(f"Directory {dir_path} created successfully")



# Function to generate ordinary commands
def get_ordinary_cmd(alpha=0.5):
	global alphas
	for dataset in DATA:
		for algo in algos:
			for sampling_rate in sampling_rates:
				if algo != "APFL":
					cmd = f"run {dataset} {algo} --sampling_rate {sampling_rate} --minibatch" 
					commands.append(cmd)
				else:
					for alpha in alphas:
						cmd = f"run {dataset} APFL --sampling_rate {sampling_rate} --alpha {alpha} --adaptive --minibatch"
						commands.append(cmd)
	return commands

# Function to generate proximal commands
def get_prox_cmd():
	global commands
	for dataset in DATA:
		for algo in algos:
			for mu in mus:
				for sampling_rate in sampling_rates:
					if algo == "L2SGD":
						cmd = f"run {dataset} {algo} --sampling_rate {sampling_rate} --comm_prob {comm_prob} --mu {mu} --minibatch"
						commands.append(cmd)
					else:
						for comm_prob in comm_probs:
							cmd = f"run {dataset} {algo} --sampling_rate {sampling_rate} --mu {mu} --minibatch"
							commands.append(cmd)
	return commands


# Function to generate fuzzy commands
def get_fuzzy_cmd():
	min_m=1.5
	algo = "FuzzyFL"
	for pre_rounds in pre_rounds_list:
		for fuzzy_m_scheduler in fuzzy_m_schedulers:
			for trans in trans_list:
				for fuzzy_m_momentum in fuzzy_m_momentums:
					for m_value in fuzzy_m_list:
						for measurement in measurements:
							if fuzzy_m_scheduler == "cosine_annealing":
								if m_value < 1.8:
									min_m = fuzzy_m_momentum-0.1
								elif m_value < 2.3:
									min_m = fuzzy_m_momentum-0.2
								else:
									min_m = fuzzy_m_momentum-0.3
							for dataset in DATA:
								for sampling_rate in sampling_rates:
									cmd = f"run {dataset} {algo} --sampling_rate {sampling_rate} --pre_rounds {pre_rounds} --fuzzy_m {m_value} --trans {trans} --min_fuzzy_m {min_m} --fuzzy_m_scheduler {fuzzy_m_scheduler} --fuzzy_m_momentum {fuzzy_m_momentum} --measurement {measurement} --minibatch"
									commands.append(cmd)
	return commands


# Example usage
# parameters_example = {"pre_rounds": "50", "fuzzy_m": "1.8"}
# inner_dir = set_inner_dir("pre_rounds", parameters_example)
# dir_path = f"dataset/algorithm/log_type/{inner_dir}"
# handle_directory(dir_path)
