import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import gaussian_filter1d
import itertools

# DATA=["emnist", "emnist_c4", "femnist", "emnist_pathologic_cl20",   "cifar100_s0.25" "cifar10_n50",   "cifar100_n20"]
# DATA = ["emnist", "emnist_c4", "femnist", "emnist_pathologic_cl20"]

TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric",
    "Local_Test/Metric",
    "Local_Test/Loss",
]

FILES_NAMES = {
    "Train/Loss": "train-loss",
    "Train/Metric": "train-acc",
    "Test/Loss": "test-loss",
    "Test/Metric": "test-acc",
    "Local_Test/Metric": "local-test-acc",
    "Local_Test/Loss": "local-test-loss",
}

AXE_LABELS = {
    "Train/Loss": "Train Loss",
    "Train/Metric": "Train Acc",
    "Test/Loss": "Test Loss",
    "Test/Metric": "Test Acc",
    "Local_Test/Metric": "Local Test Acc",
    "Local_Test/Loss": "Local Test Loss",
}
# LEGEND = {
#     "local": "Local",
#     "clustered": "Clustered FL",
#     "FedAvg_lr_0.1": "FedAvg",
#     "FedEM": "FedEM (Ours)",
#     "FedAvg_adapt": "FedAvg+",
#     "personalized": "pFedMe",
#     "FedProx": "FedProx",
#     "FuzzyFL_lr_0.03": "FuzzyFL",
# }
LEGEND = {
    "FedAvg": "FedAvg",  # "FedEM": "FedEM",
    "FuzzyFL": "FuzzyFL",
    "FedProx": "FedProx",
    "pFedMe": "pFedMe",
    "clustered": "Clustered FL",
    "APFL": "APFL",
    "L2SGD": "L2SGD",
}

MARKERS = {
    "local": "x",
    "clustered": "s",
    "FedAvg": "h",
    "FedEM": "d",
    "APFL": "q",
    "AGFL": "p",
    "FuzzyFL": "s",
    "FedAvg_adapt": "4",
    "pFedMe": "X",
    "DEM": "|",
    "FedProx": "p",
}

COLORS = {
    "local": "tab:blue",
    "clustered": "tab:orange",
    "FedAvg": "tab:green",
    "FedEM": "tab:red",
    "FedAvg_adapt": "tab:purple",
    "pFedMe": "tab:brown",
    "DEM": "tab:pink",
    "APFL": "tab:pink",
    "FedProx": "tab:cyan",
    "FuzzyFL": "tab:red",
}

PROX_METHODS = ["L2SGD", "FedProx", "pFedMe"]
# TWO_DIR_METHODS = PROX_METHODS + ["FuzzyFL", "APFL","FedProx","FedEM","FedAvg","clustered"]
ONE_DIR_METHODS=["FedAvg","FedEM"]

def empty_lst(lst):
    if len(lst) == 0:
        return True
    else:
        for string in lst:
            if len(string) > 0:
                return False
        return True


def check_mu(folder_name):
    global mu_word
    if mu_word:
        if mu_word not in folder_name:
            return False
    return True

def parse_string_to_dict(input_string):
    # 使用split方法按照下划线分割字符串
    parts = input_string.split('_')
    
    # 将分割后的结果按键-值对进行组合
    dict_ = {}
    for i in range(0, len(parts) - 1, 2):  # 步进为2，因为每两个部分为一组键-值对
        dict_[parts[i]] = parts[i+1]
    
    return dict_
    
def check_lr_samp(first_dir):
    params_dict=parse_string_to_dict(first_dir)
    # print("params_dict ",params_dict)
    if not LEARNING_RATES and not SAMPLING_RATES:  # Both lists are empty
        return True
    # if "seed" not in params_dict:
    #     return False 
    if params_dict["lr"]==learning_rate and params_dict["samp"] == SAMPLING_RATES[0]:
        return True
    return False

def get_figure_folder_name(lst):
    folder_name = ""
    for string in lst:
        parts = string.split("_")
        if folder_name != "":
            folder_name += "_"
        folder_name += "".join(parts)
    # print(folder_name)
    folder_name = folder_name.replace("sch", "sch_")
    # print(folder_name)
    return folder_name


def replace_list(lst, _strs):
    for _str in _strs:
        if _str in lst:
            lst.remove(_str)
            lst.append(_str)


def check_fuzzy_folder(folder_name):
    global FUZZY_CONTAIN_WORDS
    # print(FUZZY_CONTAIN_WORDS)
    parts = folder_name.split("_")
 
    if empty_lst(FUZZY_CONTAIN_WORDS):
        return check_folder(folder_name)
    else:
        # print("folder_name ", folder_name)
        parameters = ["_".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
        # print("fuzzy parameters", parameters)
        if (all(word in parameters for word in FUZZY_CONTAIN_WORDS) and not any(word in parameters for word in FILTER_WORDS)):
            # print("2parameters", parameters)
            return True
    return False


def check_apfl(folder_name):
    parts = folder_name.split("_")
    if empty_lst(APFL_CONTAIN_WORDS):
        return check_folder(folder_name)
    else:
        parameters = ["_".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
        if (all(word in parameters for word in APFL_CONTAIN_WORDS)
                and not any(word in parameters for word in FILTER_WORDS)):
            return True
    return False

def check_fedem(folder_name):
    parts = folder_name.split("_")
    if empty_lst(FEDEM_CONTAIN_WORDS):
        return check_folder(folder_name)
    else:
        parameters = ["_".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
        if (all(word in parameters for word in FEDEM_CONTAIN_WORDS)
                and not any(word in parameters for word in FILTER_WORDS)):
            return True
    return False

def check_folder(folder_name):
    # print("folder_name ", folder_name)
    parameters = parse_string_to_dict(folder_name)
    # print("parameters", parameters)
    if not any(word in parameters for word in FILTER_WORDS):
            return True
    return False


def is_plot_visible(method_name):
    if method_name in VISBLE_METHODS and method_name not in INVISBLE_METHODS:
        return True
    return False


def extract_event_data(method_path, tag_, task_dir):
    mode = tag_.split('/')[0].lower()
    if mode == "local_test":
        mode = "test"
    task_path = os.path.join(method_path, mode, task_dir)
    # print("task_path  ", task_path)
    if not os.path.exists(task_path):
        print(f"Directory {task_path} does not exist.")
        return None, None
    try:
        ea = EventAccumulator(task_path).Reload()
        tag_values = []
        steps = []
        for event in ea.Scalars(tag_):
            tag_values.append(event.value)
            steps.append(event.step)
        return steps, tag_values
    except KeyError:
        print(f"Tag '{tag_}' not found in {task_path}. Skipping...")
        return None, None


def extract_label(param_dir):
    # print("extract_label param_dir: ",param_dir)
    # param_dir = param_dir.replace("clustered", "clustered FL")
    # param_dir = param_dir.replace("lr_0.1_", "")
    param_dir = param_dir.replace("_trans_0.75", "")
    # param_dir = param_dir.replace("_pre_25", "")
    param_dir = param_dir.replace("pre_1", "")
    param_dir = param_dir.replace("samp_0.5", "")
    param_dir = param_dir.replace("sch_constant", "")
    param_dir = param_dir.replace("mt_0.8", "")
    param_dir = param_dir.replace("msu_euclid", "")
    # param_dir = param_dir.replace("mu_", "mu")
    minilabel = param_dir.split("_")
    # print(minilabel)
    label = " ".join(minilabel)
    # print(label)
    # print("label  ", label)
    return label


def handle_method(ax, method_dir, tag_, task_dir):
    method = method_dir.split(os.path.sep)[-1]
    # print("method ", method)
    for param_dir in os.listdir(method_dir):
        if not check_lr_samp(param_dir):
            # print("sampling rate or learning rate pass...")
            continue
        if not check_folder(param_dir):
            print("check_folder pass...")
            continue
        if os.path.isfile(os.path.join(method_dir, param_dir)):
            continue
        param_path = os.path.join(method_dir, param_dir)
        # print("param_path ", param_path)
        label = method + " " + extract_label(param_dir)
        # print("label ", label)
        steps, tag_values = extract_event_data(param_path, tag_, task_dir)
        if steps is not None:
            ax_plot(ax, steps, tag_values, label)


def handle_2dir_method(ax, method_dir, tag_, task_dir):
    method = method_dir.split(os.path.sep)[-1]
    # print("method", method)
    for sampling_dir in os.listdir(method_dir):
        if not check_lr_samp(sampling_dir):
            # print("sample or learning rate pass...")
            continue
        param_dirs = os.path.join(method_dir, sampling_dir)
        for param_dir in os.listdir(param_dirs):
            # print("param_dir ", param_dir)
            if os.path.isfile(os.path.join(param_dirs, param_dir)):
                continue
            if method in PROX_METHODS:
                if not check_mu(param_dir):
                    print("mu pass...")
                    continue
            if method == "APFL":
                if not check_apfl(param_dir):
                    print("apfl pass...")
                    continue
            if method == "FedEM":
                if not check_fedem(param_dir):
                    print("fedem learners pass...")
                    continue
            if method == "FuzzyFL":
                if not check_fuzzy_folder(param_dir):
                    # print("fuzzy pass...")
                    continue
            param_path = os.path.join(param_dirs, param_dir)
            # print("param_dir ", param_dir)
            # print("param_path ", param_path)
            steps, tag_values = extract_event_data(param_path, tag_, task_dir)
            label = method +' '+ extract_label(sampling_dir)+" " + extract_label(param_dir)
            # print("label ", label)
            if steps is not None:
                ax_plot(ax, steps, tag_values, label)


def make_plot(path_, tag_, task_dir, save_path):
    fig, ax = plt.subplots(figsize=(36, 30))
    dataset = path_.split(os.path.sep)[-1]
    mode = tag_.split('/')[0].lower()
    print("mode  ", mode)
    
    if "c10" in dataset and "FedEM" in ONE_DIR_METHODS:
        ONE_DIR_METHODS.remove("FedEM")
    if task_dir != "global" and mode == "local_test":
        return
    if mode == "local_test":
        task_dir = "client_avg"

    for method in os.listdir(path_):
        if not is_plot_visible(method):
            print(method)
            print(is_plot_visible(method))
            continue
        method_dir = os.path.join(path_, method)
        # print("\nmethod_dir ", method_dir)
        # print("TWO_DIR_METHODS", TWO_DIR_METHODS)
        if method in ONE_DIR_METHODS:
            handle_method(ax, method_dir, tag_, task_dir)
        else:
            handle_2dir_method(ax, method_dir, tag_, task_dir)

    ax.grid(True, linewidth=2)
    ax.set_ylabel(AXE_LABELS[tag_], fontsize=50)
    ax.set_xlabel("Rounds", fontsize=50)

    ax.tick_params(axis="both", labelsize=25)
    ax.legend(fontsize=40)

    os.makedirs(save_path, exist_ok=True)
    # print("save_path",save_path)
    extension = ".png"
    # print("path: " f"{dataset}_{FILES_NAMES[tag_]}{extension}")
    # print(dataset)
    # print(tag_)
    fig_path = os.path.join(save_path, f"{dataset}_{FILES_NAMES[tag_]}{extension}")

    # print("\nfig_path ", fig_path)
    title = dataset + " " + (SAMPLING_RATES[0]).replace("_", " ")
    plt.title(title, fontsize=50)
    try:
        plt.savefig(fig_path, bbox_inches="tight")
        print("Figure saved successfully.")
        plt.close()
    except Exception as e:
        print("Failed to save figure:", str(e))


def ax_plot(ax, steps, tag_values, label):
    if SMOOTH:
        tag_values = gaussian_filter1d(tag_values, sigma=1.5)

    # print("steps",len(steps))
    # print("tag_values",len(tag_values))
    ax.plot(
        steps,
        tag_values,
        linewidth=3.5,
        label=label,
    )


def plot(datasets):
    current_dir = os.getcwd()
    for dataset in datasets:
        print(dataset)
        print("learning_rate ",learning_rate)
        # relative_path = os.path.join("logs", dataset)
        relative_path = os.path.join("logs", dataset)

        path = os.path.join(current_dir, relative_path)
        inder_dir = ""
        inder_dir = "lr"+learning_rate+'_'+"samp"+SAMPLING_RATES[0]
        # inder_dir += "/Fuzzy"
        inder_dir2 = "/" + get_figure_folder_name(FUZZY_CONTAIN_WORDS)
        global mu_word
        if mu_word != "":
            # mu_word = mu_word.replace("_", "")
            inder_dir += "/" + mu_word.replace("_", "")
        print("inder_dir ",inder_dir)
        print("inder_dir2 ",inder_dir2)
         
        # inder_dir2 += ("/" + task_dir)
        pre_dir=""
        # pre_dir="_"+ORIGIN_FUZZY_CONTAIN_WORDS[0]
        # save_path = f"./figures/{dataset}/{inder_dir}{inder_dir2}"
        save_path = f"./figures_topk/{dataset}/{inder_dir}{inder_dir2}"
        for tag in TAGS:
            print("\ntag ", tag)
            make_plot(path_=path, tag_=tag, task_dir=task_dir, save_path=save_path)

            print(f"\nPlotting completed, saved in {save_path}")


if __name__ == "__main__":
    datasets = [
        # "cifar10",
        # "cifar10_alpha0.8",
        # "cifar10_pathologic_cl3",
        # "cifar100",
        # "emnist",
        # "emnist_alpha0.2",
        # "emnist_alpha0.6",
        # "emnist_alpha0.8",
        # "emnist_pathologic_cl5",
        # "emnist_pathologic_cl10",
        # "emnist_pathologic_cl20",
        # "seed23456_emnist",
        # "seed23456_emnist_alpha0.2",
        # "seed23456_emnist_alpha0.6",
        # "seed23456_emnist_alpha0.8",
        # "seed23456_emnist_pathologic_cl5",
        # "seed23456_emnist_pathologic_cl10",
        # "seed23456_emnist_pathologic_cl20",
        # "emnist_c5",
        # "emnist_c5_alpha0.2",
        # "emnist_c5_alpha0.4",
        # "emnist_c5_alpha0.6",
        # "emnist_c5_alpha0.8",
        # "emnist_c10_alpha0.2",
        "emnist_c10_alpha0.3",
        # "emnist_c10_alpha0.4",
        # "emnist_c10_alpha0.5",
        # "emnist_c10_alpha0.6",
        # "emnist_n200_c10_alpha0.3",
        # "emnist_n200_c10_alpha0.4",
        # "femnist"
    ]
    # datasets = "synthetic00"
    # VISBLE_METHODS = ["FedAvg", "L2SGD", "FedEM", "FuzzyFL", "APFL", "clusterd", "FedAvg+", "pFedMe"]
    VISBLE_METHODS = [
        "FedAvg",
        # "L2SGD",
        # "FedProx",
        # "FedEM",
        "FuzzyFL",
        # "FedSoft",
        # "APFL",
        # "clusterd",
        # "FedAvg+",
        # "pFedMe",
    ]
    INVISBLE_METHODS = []
    SMOOTH = True
    # FILTER_WORDS=["samp_0.1", "pre_1", "samp_0.2","samp_1", "sch_cosine", "sch_constant", "mt_0.5" "pre_50"]
    FILTER_WORDS = [
        "pre_50",  
        # "sch_constant",
        "sch_cosine",  
        "seed_2345",
        # "msu_euclid",
        # "msu_loss",
        "level", "alpha_0.5", "alpha_0.75", "grad"]
    # FILTER_WORDS = []
    mu_word = ""
    mu_words = [
        # "",
        "mu_0.05",
        # "mu_0.1",
        # "mu_0.2",
        # "mu_0.5"
    ]
    SAMPLING_RATES = ["0.5"]
    LEARNING_RATES = ["0.02", "0.05", "0.1"]
    mt_list = [
        # "",
        # "mt_0.5",
        "mt_0.8",
        # "mt_0"
    ]
    ORIGIN_FUZZY_CONTAIN_WORDS = [
        # "pre_50",
        "clusters_10",
        "top_5",
        # "constant"
        ]
    m_list = [
        # [],
        # ["m_1.2"],
        # ["m_1.3"],
        # ["m_1.4"],        
        ["m_1.5"],
        ["m_1.6"],
        ["m_1.7"],
        ["m_1.8"],
        ["m_2"],
        ["m_2.2"],
        # ["m_2.4"],
        # ["m_2.6"],
        # ["m_2.8"]
        # ["m_1.9"],
    ]

    # print(FUZZY_CONTAIN_WORDS_list)
    trans_list = [
        # "",
        # "trans_0.5",
        "trans_0.75",
        # "trans_0.9"
        # "trans_0"
    ]
    APFL_CONTAIN_WORDS = ["adaptive"]
    FEDEM_CONTAIN_WORDS= []
    FUZZY_CONTAIN_WORDS_list = []
    combinations = list(itertools.product(mt_list, m_list, trans_list))
    print(combinations)
    for combination in combinations:
        words = []
        for sublist in combination:
            if isinstance(sublist, list) and sublist:
                words += sublist
            elif sublist:
                words.append(sublist)
        FUZZY_CONTAIN_WORDS_list.append(words)

    print(FUZZY_CONTAIN_WORDS_list)

    task_dir = "global"
    for FUZZY_CONTAIN_WORDS in FUZZY_CONTAIN_WORDS_list:
        # print("FUZZY_CONTAIN_WORDS ", FUZZY_CONTAIN_WORDS)
        FUZZY_CONTAIN_WORDS = ORIGIN_FUZZY_CONTAIN_WORDS+FUZZY_CONTAIN_WORDS
        print("FUZZY_CONTAIN_WORDS ", FUZZY_CONTAIN_WORDS)
        for learning_rate in LEARNING_RATES:
            for mu_word in mu_words:
                plot(datasets)
    # for i in range(5):
    #     task_dir = f"task_{i}"
    #     print(task_dir)
    #     plot(datasets)
