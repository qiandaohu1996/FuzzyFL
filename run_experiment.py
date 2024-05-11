import csv
import sys
from utils.utils import *
from utils.constants import *
from utils.args import *
from utils.my_profiler import *
import time
import os
from rich.progress import track
# from torch.utils.tensorboard import SummaryWriter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

pb = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(bar_width=100), 
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)
def run_experiment(args_):
    torch.manual_seed(args_.seed)
    data_dir = os.path.join('data', args_.experiment, 'all_data')

    logs_root = args_.logs_root if "logs_root" in args_ else get_logdir(args_)

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

    header=["Round", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"]
    logs_dir = os.path.join(logs_root, "train")
    os.makedirs(logs_dir, exist_ok=True)
    train_client_global_logger_path = os.path.join(logs_dir, "global.csv")
    train_client_global_logger = CSVLogger()
    train_client_global_logger.openfile(train_client_global_logger_path, 'w', newline='')
    train_client_global_logger.writerow(header)

    logs_dir = os.path.join(logs_root, "test")
    os.makedirs(logs_dir, exist_ok=True)
    test_client_global_logger_path = os.path.join(logs_dir, "global.csv")
    test_client_global_logger = CSVLogger()
    test_client_global_logger.openfile(test_client_global_logger_path, 'w', newline='')
    test_client_global_logger.writerow(header)

    # Initialization and aggregation code remains the same
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
                tau=args_.tau,
                top=args_.top,
                pre_rounds=args_.pre_rounds,
                log_freq=args_.log_freq,
                train_client_global_logger=train_client_global_logger,
                test_client_global_logger=test_client_global_logger,
                test_clients=test_clients,
                single_batch_flag=args_.minibatch,
                verbose=args_.verbose,
                seed=args_.seed
            )
    # pbar = tqdm(total=args_.n_rounds, dynamic_ncols=True)
    print("\n========Training begins========\n")
    for _ in track(range(args_.n_rounds+1), description="Training..."):
        aggregator.mix()
        if aggregator.c_round % 200 == 0:
            # Saving model state and CSV logs
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
    
    print(f"\nexecution time {exec_time}")
