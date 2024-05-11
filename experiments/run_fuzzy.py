import subprocess
from multiprocessing import Process, Manager
from itertools import product

def params_to_cmd(params):
    cmd=[]
    for key, value in params.items():
        if key in ['dataset', 'method']:
            cmd.append(value)
        elif isinstance(value,bool) and value==True:
            cmd.append("--" + key)
        else:
            cmd.extend(["--" + key, str(value)])
    return cmd     
  
def get_min_m(m):
    if m < 1.8:
        subtract =  0.1
    elif m < 2.3:
        subtract =  0.2
    else:
        subtract =  0.3
    result = m - subtract
    return int(result) if result == int(result) else round(result, 1)

def get_fuzzy_commands():
    commands = []
    for combo in product(datasets, learner_rates, sampling_rates, pre_rounds_list, fuzzy_m_list, fuzzy_m_momentums, fuzzy_m_schedulers, trans_list, measurements):
        dataset, lr, sampling_rate, pre_rounds, m, fuzzy_m_momentum, fuzzy_m_scheduler, trans, measurement = combo
        min_m=1.2
        if fuzzy_m_scheduler == "cosine_annealing":
            min_m = get_min_m(m)
                
        dict_ = {
            'dataset': dataset,
            'method': "FuzzyFL",
            'lr': lr,
            'n_rounds': n_rounds,
            'sampling_rate': sampling_rate,
            'pre_rounds': pre_rounds,
            'n_clusters': n_clusters,
            'top': top,
            'fuzzy_m': m,
            'trans': trans,
            'min_fuzzy_m': min_m,
            'fuzzy_m_scheduler': fuzzy_m_scheduler,
            'fuzzy_m_momentum': fuzzy_m_momentum,
            'measurement': measurement,
        }
        
        if use_byzantine:
            dict_['use_byzantine'] = use_byzantine
            dict_['byzantine_ratio'] = byzantine_ratio
            dict_['z_max'] = z_max
        dict_.update(extra_params)
        
        cmd_args=params_to_cmd(dict_)
        
        command = ['python', 'run_experiment.py'] + cmd_args 
        commands.append(command)
        
    return commands

def run_command(cmd, idx):
    print(f"Process {idx}")
    subprocess.run(cmd, shell=True)

def run_commands(commands, max_processes=3):
    with Manager() as manager:
        idx = manager.Value('i', 0)
        processes = []
        for cmd in commands:
            if len(processes) >= max_processes:
                for process in processes:
                    process.join()
                processes.clear()
            idx.value += 1
            process = Process(target=run_command, args=(cmd, idx.value))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
if __name__ == '__main__':  
    extra_params = {
        "optimizer": "adam",
        "n_rounds": 200,
        "lr_scheduler": "multi_step",
        "bz": 128,
        "local_steps": 1,
        "log_freq": 2,
        "seed": 1234,
        "verbose": 1
    }
    max_processes=3
    # sampling_rates = [0.5, 1]
    sampling_rates = [0.5 ]
    # learner_rates = [0.02]
    learner_rates = [0.02, 0.01]
    # use_byzantines=[True,False]
    use_byzantines=[False]
    byzantine_ratio = 0.1
    z_max = 0.3
    n_rounds=200
    verbose=1
    bz=128

    # pre_rounds_list = [1]
    pre_rounds_list = [10]
    fuzzy_m_list = [1.5, 1.75, 2.0]
    fuzzy_m_momentums = [0.9]

    fuzzy_m_schedulers = ["cosine_annealing", "constant"]
    trans_list = [0.8]
    measurements = ["euclid" 'loss']
    n_clusters=3
    top=3
    use_byzantine=False

    datasets=[
        # 'emnist',
        # 'femnist',
        # 'emnist_alpha0.2',
        # 'emnist_alpha0.6',
        'emnist_c5',
        # 'emnist_pathologic_cl5',
        # 'emnist_pathologic_cl10',
        'emnist_pathologic_cl20',
        # 'cifar10',
        ]
        
    commands = get_fuzzy_commands()
    print(len(commands))

    # for cmd in commands:
    #     print(cmd)
    run_commands(commands)