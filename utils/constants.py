"""Configuration file for experiments"""
import string


ALL_STRATEGIES = {"random"}

ALL_MODELS = {"mobilenet"}

DEFAULT_LOADER_TYPE = "tabular"
DEFAULT_EXTENSION = ".pkl"
SYNTHETIC_LIST = ["synthetic", "synthetic00", "synthetic01"]

SPECIAL_EXTENSIONS = {
    "femnist": ".pt",
    "seed23456_femnist": ".pt",
    "shakespeare": ".txt",
    "shakespeare_s0.3": ".txt",
}


def get_loader_type(data_type):
    return DEFAULT_LOADER_TYPE if data_type in SYNTHETIC_LIST else data_type


def get_extension(data_type):
    return SPECIAL_EXTENSIONS.get(data_type, DEFAULT_EXTENSION)


LOADER_TYPE = [
    "synthetic", "cifar10", "cifar100", "cifar100_s0.25", "emnist", "emnist_pathologic", "emnist_pathologic_cl10",
    "emnist_pathologic_cl20", "emnist_component4", "emnist20", "femnist", "shakespeare"
]

EMINST_LOADER_LIST = [
    "emnist",
    "emnist20",
    "emnist_c5",
    "emnist_c5_n50",
    "emnist_n25",
    "emnist_c5_topk",
    "emnist_c5_alpha0.2",
    "emnist_c5_alpha0.4",
    "emnist_c5_alpha0.6",
    "emnist_c5_alpha0.8",
    "emnist_c10_alpha0.1",
    "emnist_c10_alpha0.2",
    "emnist_c10_alpha0.3",
    "emnist_c10_alpha0.4",
    "emnist_c10_alpha0.5",
    "emnist_c10_alpha0.6",
    "emnist_n200_c10_alpha0.3",
    "emnist_n200_c10_alpha0.4",
    "emnist_c10_topk",
    "emnist_alpha0.2",
    "emnist_alpha0.6",
    "emnist_alpha0.61",
    "emnist_alpha0.8",
    "emnist20_c4",
    "emnist_pathologic_cl5",
    "emnist_pathologic_cl10",
    "emnist_pathologic_cl20",
    "emnist_cluster10",
    "seed23456_emnist",
    "seed23456_emnist20",
    "seed23456_emnist_c5",
    "seed23456_emnist_alpha0.2",
    "seed23456_emnist_alpha0.6",
    "seed23456_emnist_alpha0.61",
    "seed23456_emnist_alpha0.8",
    "seed23456_emnist20_c4",
    "seed23456_emnist_pathologic_cl5",
    "seed23456_emnist_pathologic_cl10",
    "seed23456_emnist_pathologic_cl20",
]
FEMNIST_LIST = ["femnist", "seed23456_femnist", "femnist179"]

EMINST_LERANER_LIST = EMINST_LOADER_LIST + FEMNIST_LIST

SYNTHETIC_LIST = ["synthetic", "synthetic00", "synthetic01"]
CIFAR10_LIST = ["cifar10","seed23456_cifar10", "cifar10_pathologic_cl3", "cifar10_alpha0.8"]
CIFAR100_LIST = ["cifar100","seed23456_cifar100", "cifar100_s0.25"]
SHAKESPEARE_LIST = ["shakespeare", "shakespeare_s0.3"]

# EXTENSIONS = {
#     "tabular": ".pkl",
#     "cifar10": ".pkl",
#     "cifar100": ".pkl",
#     "emnist": ".pkl",
#     "emnist_pathologic": ".pkl",
#     "emnist_pathologic_cl10": ".pkl",
#     "emnist_pathologic_cl20": ".pkl",
#     "emnist_component4": ".pkl",
#     "emnist20": ".pkl",
#     "femnist": ".pt",
#     "shakespeare": ".txt",
# }

AGGREGATOR_TYPE = {
    "FedEM": "centralized",
    "FedAvg": "centralized",
    "FedProx": "centralized",
    "local": "no_communication",
    "pFedMe": "personalized",
    "clustered": "clustered",
    "FuzzyFL": "FuzzyFL",
    "FedSoft": "FedSoft",
    "APFL": "APFL",
    "AGFL": "AGFL",
    "L2SGD": "L2SGD",
    "AFL": "AFL",
    "FFL": "FFL",
    "FedAvgDecentralized": "FedAvgDecentralized",
    "FuzzyDecentralized": "FuzzyDecentralized"
}

CLIENT_TYPE = {
    "FedEM": "mixture",
    "AFL": "AFL",
    "FFL": "FFL",
    "APFL": "normal",
    "AGFL": "normal",
    "FuzzyFL": "FuzzyFL",
    "FedSoft": "FedSoft",
    "L2SGD": "normal",
    "FedAvg": "normal",
    "FedProx": "normal",
    "local": "normal",
    "pFedMe": "normal",
    "clustered": "normal",
    "FedAvgDecentralized": "FedAvgDecentralized",
    "FuzzyDecentralized": "FuzzyDecentralized"
}

SHAKESPEARE_CONFIG = {
    "input_size": len(string.printable),
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": len(string.printable),
    "n_layers": 2,
    "chunk_len": 80
}

CHARACTERS_WEIGHTS = {
    '\n': 0.43795308843799086,
    ' ': 0.042500849608091536,
    ',': 0.6559597911540539,
    '.': 0.6987226398690805,
    'I': 0.9777491725556848,
    'a': 0.2226022051965085,
    'c': 0.813311655455682,
    'd': 0.4071860494572223,
    'e': 0.13455606165058104,
    'f': 0.7908671114133974,
    'g': 0.9532922255751889,
    'h': 0.2496906467588955,
    'i': 0.27444893060347214,
    'l': 0.37296488139109546,
    'm': 0.569937324017103,
    'n': 0.2520734570378263,
    'o': 0.1934141300462555,
    'r': 0.26035705948768273,
    's': 0.2534775933879391,
    't': 0.1876471355731429,
    'u': 0.47430062920373184,
    'w': 0.7470615815733715,
    'y': 0.6388302610200002
}

LOCAL_HEAD_UPDATES = 10                    # number of epochs for local heads used in FedRep

# NUM_WORKERS = os.cpu_count()  # number of workers used to load data and in GPClassifier
NUM_WORKERS = 1
