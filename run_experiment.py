"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import sys
from utils.utils import *
from utils.constants import *
from utils.args import *
import time
import os
from torch.utils.tensorboard import SummaryWriter


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    with segment_timing("init clients "):

        clients = init_clients(
            args_, root_path=os.path.join(data_dir, "train"), logs_root=os.path.join(logs_root, "train")
        )
    print("Clients number ", len(clients))

    print("==> Test Clients initialization..")
    with segment_timing("init test clients"):

        test_clients = init_clients(
            args_, root_path=os.path.join(data_dir, "test"), logs_root=os.path.join(logs_root, "test")
        )
    print("test_clients number ", len(test_clients))

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "client_avg")
    os.makedirs(logs_path, exist_ok=True)
    local_test_logger = SummaryWriter(logs_path)

    # with segment_timing("init global_learners_ensemble"):

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            method=args_.method,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            alpha=args_.alpha,
            adaptive_alpha=args_.adaptive_alpha,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu

        )
    # with segment_timing("init aggregator"):
    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]
        aggregator =\
            get_aggregator(
                aggregator_type=aggregator_type,
                clients=clients,
                global_learners_ensemble=global_learners_ensemble,
                lr_lambda=args_.lr_lambda,
                lr=args_.lr,
                q=args_.q,
                mu=args_.mu,
                fuzzy_m_momentum=args_.fuzzy_m_momentum,
                measurement=args_.measurement,
                comm_prob=args_.comm_prob,
                sampling_rate=args_.sampling_rate,
                n_clusters=args_.n_clusters,
                pre_rounds=args_.pre_rounds,
                log_freq=args_.log_freq,
                global_train_logger=global_train_logger,
                global_test_logger=global_test_logger,
                local_test_logger=local_test_logger,
                test_clients=test_clients,
                single_batch_flag=args_.minibatch,
                verbose=args_.verbose,
                seed=args_.seed
            )
    torch.cuda.empty_cache()
    # print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0

    print("\n========Training begins======== \n", time.ctime())
    with segment_timing("Training total time "):
        while current_round <= args_.n_rounds:

            print(f"===========train at {current_round} round===========")
            with segment_timing(f"{current_round} round executing time "):
                aggregator.mix()

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

            if aggregator.c_round % 200 == 0:
                if "save_path" in args_:
                    save_root = os.path.join(args_.save_path) + f"{aggregator.c_round}round"
                    os.makedirs(save_root, exist_ok=True)
                    aggregator.save_state(save_root)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # torch._dynamo.config.cache_size_limit = 64
    torch.backends.cudnn.benchmark = True                    # 启用快速卷积算法
    # torch.backends.cudnn.deterministic = True                    # 使用确定性卷积算法
    command = ' '.join(sys.argv)
    print("command line: ", command)

    start_time = time.time()
    print("start at ", time.ctime())

    args = parse_args()
    run_experiment(args)

    end_time = time.time()

    print("end at ", time.ctime())

    print("command line: ", command)

    exec_time = end_time - start_time
    exec_time = time.strftime('%H:%M:%S', time.gmtime(exec_time))
    # 打印格式化后的时间
    print(f"\nexecution time {exec_time}")
