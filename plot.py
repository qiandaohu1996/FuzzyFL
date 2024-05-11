
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from itertools import product
from pathlib import Path

from experiments.run_fuzzy import get_min_m

datasets=[
    # 'emnist',
    # 'femnist',
    # 'emnist_alpha0.2',
    # 'emnist_alpha0.6',
    # 'emnist_c5',
    # 'emnist_pathologic_cl5',
    # 'emnist_pathologic_cl10',
    'emnist_pathologic_cl20',
    # 'cifar10',
    ]

samps = [0.5]  # samp
lrs = [0.02, 0.01]  # lr
pres = [10, 50]  # pre_rounds
fschs = ["cosine" ]  # sch
fms = [1.5, 1.75, 2.0]  # fuzzy_m
fmts = [0.9]  # fuzzy_m_momentums
fmsus = ["euclid" ]  # measurements
trans_list = [0.8]  # trans

logs_root = Path('logs')
figs_root = Path('figs')

all_paths = []
mu=0.2
methods = ['FuzzyFL', 'FedAvg', 'FedEM']
        

def get_method_logdir(method, dataset, first_dir, inner_dir):
    if method=='FedAvg':
        inner_dir=''
    if method=='FedEM':
        inner_dir='nlearners_3'
    if method=='FedProx':
        inner_dir= f'mu_{mu}'
    logdir = logs_root / dataset / method / first_dir / inner_dir
    return logdir

def get_fuzzy_innerdirs():
    innerdirs=[]
    inner_figdirs=[]
    for combo in product(pres, fschs, fms, fmts, fmsus, trans_list):
        pre, sch, m, mt, msu, trans=combo 
        minm = get_min_m(m)  
        innerdir = f'pre_{pre}_sch_{sch}_minm_{minm}_m_{m}_msu_{msu}_trans_{trans}_mt_{mt}_clusters_3'
        innerdirs.append(innerdir)
        inner_figdir= f'pre_{pre}/sch_{sch}_m_{m}_msu_{msu}'
        inner_figdirs.append(inner_figdir)
    return innerdirs,inner_figdirs


def get_firstdirs(samps,lrs):
    first_dirs=[]
    for samp, lr in product(samps, lrs):
        first_dir = f'lr_{lr}_samp_{samp}'
        first_dirs.append(first_dir)
    return first_dirs

first_dirs=get_firstdirs(samps,lrs)
fuzzy_innerdirs,inner_figdirs=get_fuzzy_innerdirs()

def load_data(logdirs, client_types):

    for method, logdir in logdirs.items():
        if method == 'FuzzyFL' and not os.path.exists(logdir):
            return None
        dfs = {}
        for client_type in client_types:
            csv_file = os.path.join(logdir, client_type, 'global.csv')
            if os.path.exists(csv_file):
                dfs[method] = {}
                df = pd.read_csv(csv_file)
                if not df.empty:
                    dfs[method][client_type] = df
                else:
                    print(f"No data found in {csv_file}")
            else:
                print(f"File {csv_file} does not exist")
        if not dfs[method]:
            del dfs[method]
    return dfs


def plot_metrics(dfs, metric_types, client_types, figdir):
    dfs=load_data(logdirs, client_types)
    if dfs is None:
        return
    for metric_type in metric_types:
        for client_type in client_types:
            plt.figure(figsize=(15, 10))
            for method, method_df in dfs.items():
                df=method_df[client_type]
                plt.plot(df['Round'], df[metric_type], label=method)

            plt.xlabel('Round')
            plt.ylabel(metric_type)
            plt.legend()
            plt.title(f'{metric_type} for {client_type}')
            plt.tight_layout()
            client_figdir=figdir/client_type
            os.makedirs(client_figdir, exist_ok=True)
            plt.savefig(f'{client_figdir}/{metric_type}.png')
            plt.close()
            
metric_types = ['TrainLoss', 'TrainAcc', 'TestLoss', 'TestAcc']
client_types = ['train', 'test']

for dataset,first_dir in product(datasets,first_dirs):
    for (fuzzy_innerdir,inner_figdir) in zip(fuzzy_innerdirs,inner_figdirs):
        logdirs={}
        for method in methods:
            logdirs[method]=get_method_logdir(method, dataset, first_dir, fuzzy_innerdir)
        figdir = figs_root / dataset / first_dir / inner_figdir 

        plot_metrics(logdirs,metric_types, client_types,figdir)
