import subprocess
from itertools import product
from multiprocessing import Process, Manager


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

def get_whole_cmd(params):
    if params['dataset'].startswith('synthetic'):
        params["input_dimension"] = 150
        params["output_dimension"] = 2
    params['n_learners']= 3 if params['method']=='FedEM' else 1
        
    prox_methods=["FedProx", "pFedMe"]
    optimizer = "prox_sgd" if params['method'] in prox_methods else "adam"

    params.update({"optimizer": optimizer})
    params.update(extra_params)
    cmd_args=params_to_cmd(params)
    command = ['python', 'run_experiment.py'] + cmd_args 
    
    return command


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