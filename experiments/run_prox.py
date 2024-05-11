from itertools import product
# sys.path.append('experiments')
from uilts import get_whole_cmd, run_commands


def get_prox_cmd():
    commands = []
    for combo in product(methods, datasets, sampling_rates, learner_rates, mus, comm_probs):
        method, dataset, sampling_rate, lr, mu, comm_prob = combo
        dict_ = {
            'dataset': dataset,
            'method': method,
            'lr': lr,
            'sampling_rate': sampling_rate,
            'mu': mu,
        }
        if method == "L2SGD":
            dict_['comm_prob'] = comm_prob
        if use_byzantine :
            dict_['use_byzantine'] = True
            dict_['byzantine_ratio'] = byzantine_ratio
            dict_['z_max'] = z_max
        command=get_whole_cmd(dict_)
        commands.append(command)
    return commands

            
if __name__ == '__main__':  

    # sampling_rates = [0.5, 1]
    sampling_rates = [0.5 ]
    # learner_rates = [0.01]
    # learner_rates = [0.02]
    learner_rates = [0.05, 0.02, 0.01]
    
    mus=[0.1 ]
    comm_probs=[0.5]
    
    # use_byzantines=[True,False]
    use_byzantines=[False]
    use_byzantine=False
    byzantine_ratio = 0.1
    z_max = 0.3
    
    datasets=[
        'emnist',
        # 'femnist',
        # 'emnist_alpha0.2',
        # 'emnist_alpha0.6',
        # 'emnist_c5',
        # 'emnist_pathologic_cl5',
        # 'emnist_pathologic_cl10',
        # 'emnist_pathologic_cl20',
        # 'cifar10',
        ]
        
    methods = ['FedProx']
    # methods = ['FedAvg']
    commands=get_prox_cmd()
    print(len(commands))
    for cmd in commands:
        print(cmd)
    run_commands(commands,2)