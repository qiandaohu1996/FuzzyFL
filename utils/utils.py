import csv
import torch
import time
import sys

from utils.csvlogger import CSVLogger
from utils.decentralized import get_mixing_matrix
sys.path.append("./")

import random

from client import *
from server import *
from server.FuzzyAgg import FuzzyGroupAggregator
from client.FuzzyClient import FuzzyClient

from client.FedSoftClient import FedSoftClient
from client.FuzzyDecentralizedClient import FuzzyDecentralizedClient
from server.FuzzyDecentralizedSystem import FuzzyDecentralizedSystem

from utils.fuzzy_utils import get_fuzzy_m_scheduler

from utils.models import *
from utils.datasets import *
from learners.learner import *
from learners.learners_ensemble import *

from .optim import *
from .metrics import *
from .constants import *
from torch.utils.data import DataLoader

from tqdm import tqdm
# import csv

             
def get_learner(
    name,
    device,
    optimizer_name,
    scheduler_name,
    initial_lr,
    mu,
    n_rounds,
    seed,
    input_dim=None,
    output_dim=None,
    is_byzantine=False,
    z_max=0.1  # 拜占庭攻击强度
):
 
    torch.manual_seed(seed)
    if name in SYNTHETIC_LIST:
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            model = LinearLayer(input_dim, 1).to(device)
            
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            
            is_binary_classification = False
    elif name in CIFAR10_LIST:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=10).to(device)
        # model = get_resnet18(n_classes=10).to(device)
        
        is_binary_classification = False
    elif name in CIFAR100_LIST:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=100).to(device)
        
        is_binary_classification = False
    elif name in EMINST_LERANER_LIST:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        
        is_binary_classification = False
    elif name in SHAKESPEARE_LIST:
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[
                character
            ]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(
            device
        )
        metric = accuracy
        model = NextCharacterLSTM(
            input_size=SHAKESPEARE_CONFIG["input_size"],
            embed_size=SHAKESPEARE_CONFIG["embed_size"],
            hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
            output_size=SHAKESPEARE_CONFIG["output_size"],
            n_layers=SHAKESPEARE_CONFIG["n_layers"],
        ).to(device)
        
        is_binary_classification = False
    else:
        raise NotImplementedError
    # model = torch.compile(model) 
    optimizer = get_optimizer(
        optimizer_name=optimizer_name, model=model, lr_initial=initial_lr, mu=mu
    )
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer, scheduler_name=scheduler_name, n_rounds=n_rounds
    )

    if name in SHAKESPEARE_LIST:
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            is_byzantine=is_byzantine,
            z_max=z_max  # 拜占庭攻击强度
        )


def get_learners_ensemble(
    n_learners,
    name,
    device,
    optimizer_name,
    scheduler_name,
    initial_lr,
    mu,
    method,
    adaptive_alpha,
    alpha,
    n_rounds,
    seed,
    input_dim=None,
    output_dim=None,
    is_byzantine=False,
    z_max=0.1  # 拜占庭攻击强度
):
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            is_byzantine=is_byzantine,
            z_max=z_max  # 拜占庭攻击强度
        )
        for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if name in SHAKESPEARE_LIST:
        return LanguageModelingLearnersEnsemble(
            learners=learners, learners_weights=learners_weights
        )
    elif method == "APFL":
        return APFLLearnersEnsemble(
            learners=learners,
            learners_weights=learners_weights,
            adaptive_alpha=adaptive_alpha,
            alpha=alpha,
        )
    elif method == "AGFL":
        return AGFLLearnersEnsemble(
            learners=learners,
            learners_weights=learners_weights,
            adaptive_alpha=adaptive_alpha,
            alpha=alpha,
        )
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_loaders(type_, root_path, batch_size, is_validation):
    if type_ in CIFAR10_LIST:
        inputs, targets = get_cifar10()
    elif type_ in CIFAR100_LIST:
        inputs, targets = get_cifar100()
    elif type_ in EMINST_LOADER_LIST:
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []
    sys.stdout.flush()
    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path), dynamic_ncols=True)):
        sys.stdout.flush()
        task_data_path = os.path.join(root_path, task_dir)
        extension = get_extension(type_)
        train_iterator = get_loader(
            type_=type_,
            path=os.path.join(task_data_path, f"train{extension}"),
            batch_size=batch_size,
            inputs=inputs,
            targets=targets,
            train=True,
        )
        val_iterator = get_loader(
            type_=type_,
            path=os.path.join(task_data_path, f"train{extension}"),
            batch_size=batch_size,
            inputs=inputs,
            targets=targets,
            train=False,
        )
        if is_validation:
            test_set = "val"
        else:
            test_set = "test"
        test_iterator = get_loader(
            type_=type_,
            path=os.path.join(task_data_path, f"{test_set}{extension}"),
            batch_size=batch_size,
            inputs=inputs,
            targets=targets,
            train=False,
        )
        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
    return train_iterators, val_iterators, test_iterators


def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ in CIFAR10_LIST:
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ in CIFAR100_LIST:
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ in EMINST_LOADER_LIST:
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ in FEMNIST_LIST:
        dataset = SubFEMNIST(path)
    elif type_ in SHAKESPEARE_LIST:
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(
            f"{type_} not recognized type; possible are {LOADER_TYPE}"
        )

    if len(dataset) == 0:
        return

    drop_last = (
        ((type_ in CIFAR100_LIST) or (type_ in CIFAR10_LIST))
        and (len(dataset) > batch_size)
        and train
    )

    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=train, drop_last=drop_last
    )
 

def init_clients(args_, root_path, logs_root):
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = get_loaders(
        type_=get_loader_type(args_.experiment),
        root_path=root_path,
        batch_size=args_.bz,
        is_validation=args_.validation,
    )

    print("===> Initializing clients..")
    clients_ = []
    total_clients = len(train_iterators)
    if args_.use_byzantine:
        byzantine_ratio = args_.byzantine_ratio  
        byzantine_count = int(byzantine_ratio * total_clients)
        byzantine_indices = random.sample(range(total_clients), byzantine_count)
        print("byzantine_count/total_clients ", f'{byzantine_count}/{total_clients}')
        print("byzantine_indices ", byzantine_indices)

    for task_id, (train_iterator, val_iterator, test_iterator) in enumerate(
        tqdm(zip(train_iterators, val_iterators, test_iterators), total=total_clients, dynamic_ncols=True)
    ):
        sys.stdout.flush()
        
        if train_iterator is None or test_iterator is None:
            continue
        
        is_byzantine = task_id in byzantine_indices if args_.use_byzantine else False
        learners_ensemble = get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            method=args_.method,
            adaptive_alpha=args_.adaptive_alpha,
            alpha=args_.alpha,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            is_byzantine=is_byzantine,  # Pass this flag to the client
            z_max=args_.z_max
        )
        os.makedirs(logs_root, exist_ok=True)
        log_path = os.path.join(logs_root, f"task_{task_id}.csv")
        
        logger = CSVLogger()
        logger.openfile(log_path, 'w', newline='')
        header=["Round", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"]
        logger.writerow(header)

        # Determine if this client should be Byzantine
        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            initial_fuzzy_m=args_.fuzzy_m,
            min_fuzzy_m=args_.min_fuzzy_m,
            fuzzy_m_scheduler_name=args_.fuzzy_m_scheduler,
            comm_prob=args_.comm_prob,
            n_rounds=args_.n_rounds,
            trans=args_.trans,
            idx=task_id,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients,
            seed=args_.seed,
        )

        clients_.append(client)
    return clients_


def get_client(
    client_type,
    learners_ensemble,
    q,
    fuzzy_m_scheduler_name,
    initial_fuzzy_m,
    min_fuzzy_m,
    n_rounds,
    trans,
    idx,
    comm_prob,
    train_iterator,
    val_iterator,
    test_iterator,
    logger,
    local_steps,
    tune_locally,
    seed 
):
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            # is_byzantine = is_byzantine,
            # z_max = z_max 
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            # is_byzantine = is_byzantine
            # z_max = z_max 
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q,
        )
    elif client_type == "FedSoft":
        return FedSoftClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )
    elif client_type == "FuzzyFL":
        fuzzy_m_scheduler = get_fuzzy_m_scheduler(
            initial_fuzzy_m=initial_fuzzy_m,
            scheduler_name=fuzzy_m_scheduler_name,
            min_fuzzy_m=min_fuzzy_m,
            n_rounds=n_rounds,
        )
        return FuzzyClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            initial_fuzzy_m=initial_fuzzy_m,
            fuzzy_m_scheduler=fuzzy_m_scheduler,
            trans=trans,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally 
        )
    elif client_type == "FuzzyDecentralized": 
        return FuzzyDecentralizedClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            initial_fuzzy_m=initial_fuzzy_m,
            comm_prob=comm_prob,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            seed=seed,
        )
    elif client_type == "FedAvgDecentralized": 
        return FedAvgDecentralizedClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            initial_fuzzy_m=initial_fuzzy_m,
            comm_prob=comm_prob,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            seed=seed,
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally 
        )


def get_aggregator(
    aggregator_type,
    clients,
    global_learners_ensemble,
    lr,
    lr_lambda,
    mu,
    comm_prob,
    q,
    sampling_rate,
    measurement,
    log_freq,
    fuzzy_m_momentum,
    n_clusters,
    top,
    tau,
    train_client_global_logger,
    test_client_global_logger,
    test_clients,
    verbose,
    pre_rounds,
    single_batch_flag,
    seed=None,
):
    seed = seed if (seed is not None and seed >= 0) else int(time.time())
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            test_clients=test_clients,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            single_batch_flag=single_batch_flag,
            seed=seed,
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            test_clients=test_clients,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            test_clients=test_clients,
            comm_prob=comm_prob,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "FuzzyDecentralized":
        return FuzzyDecentralizedSystem(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "FedAvgDecentralized":
        return FedAvgDecentralizedSystem(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            verbose=verbose,
            seed=seed,
        )
    # elif aggregator_type == "AGFL":
    #     return AGFLAggregator(
    #         clients=clients,
    #         global_learners_ensemble=global_learners_ensemble,
    #         log_freq=log_freq,
    #         pre_rounds=pre_rounds,
    #         test_clients=test_clients,
    #         train_client_global_logger=train_client_global_logger,
    #         test_client_global_logger=test_client_global_logger,
    #         comm_prob=comm_prob,
    #         sampling_rate=sampling_rate,
    #         single_batch_flag=single_batch_flag,
    #         verbose=verbose,
    #         seed=seed,
    #     )
    # elif aggregator_type == "AGFL2":
    #     return GroupAPFL(
    #         clients=clients,
    #         global_learners_ensemble=global_learners_ensemble,
    #         log_freq=log_freq,
    #         pre_rounds=pre_rounds,
    #         test_clients=test_clients,
    #         train_client_global_logger=train_client_global_logger,
    #         test_client_global_logger=test_client_global_logger,
    #         comm_prob=comm_prob,
    #         sampling_rate=sampling_rate,
    #         single_batch_flag=single_batch_flag,
    #         verbose=verbose,
    #         seed=seed,
    #     )
    elif aggregator_type == "FuzzyFL":
        return FuzzyGroupAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            pre_rounds=pre_rounds,
            n_clusters=n_clusters,
            top=top,
            fuzzy_m_momentum=fuzzy_m_momentum,
            measurement=measurement,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            comm_prob=comm_prob,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "FedSoft":
        return FedSoftAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            pre_rounds=pre_rounds,
            n_clusters=n_clusters,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            mu=mu,
            tau=tau,
            sampling_rate=sampling_rate,
            single_batch_flag=single_batch_flag,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "APFL":
        print("util ", single_batch_flag)
        return APFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` ,  `AGFL`, `FuzzyFL` and `decentralized`."
        )


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None and target is not None:
            input.record_stream(torch.cuda.current_stream())
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
