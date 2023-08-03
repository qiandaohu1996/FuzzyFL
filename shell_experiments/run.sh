#!/usr/bin/env bash
# shellcheck disable=SC2034

declare -g DATA=()
declare -g algos=()

# algorithm parameters
declare -g algo=""
declare -g lr=0.1
declare -g bz=256
declare -g local_steps=1

declare -g n_rounds=200
declare -g log_freq=2

declare -g sampling_rates=("0.5")

# for fuzzyFL
declare -g pre_rounds_list=("50")
declare -g fuzzy_m_schedulers=("constant")
declare -g fuzzy_m_list=("1.8")
declare -g min_m_list=("1.5")
declare -g min_m="1.5"
declare -g trans_list=("0.75")
declare -g fuzzy_m_momentums=("0.8")
declare -g measurements=("loss")

# for l2sgd fedprox and pfedMe
declare -g mus=("0.1" "0.5")
# for l2sgd
declare -g comm_probs=("0.2" "0.5")

# for apfl
declare -g alphas=()

# get commands
declare -g commands=()

set_inner_dir() {
    local param="$1"
    declare -A paramToTemplate
    paramToTemplate=(["pre_rounds"]="pre_%s"
        ["fuzzy_m"]="_m_%s"
        ["min_fuzzy_m"]="_minm_%s"
        ["sampling_rate"]="_samp_%s"
        ["locally_tune_clients"]="_adapt"
        ["adaptive"]="_adapt"
        ["fuzzy_m_scheduler"]="_sch_%s"
        ["measurement"]="_msu_%s"
        ["fuzzy_m_momentum"]="_mt_%s"
        ["comm_prob"]="comm_%s_"
        ["n_clusters"]="_cluster_%s"
        ["mu"]="mu_%s"
    )

    if [[ -v "parameters[$param]" ]]; then
        if [[ "${parameters[$param]}" != "true" ]]; then
            local template=${paramToTemplate[$param]}
            if [[ -z "$template" ]]; then
                template="_${param}_%s" # default template
            fi
            # shellcheck disable=SC2059
            inner_dir+=$(printf "$template" "${parameters[$param]}")

        else
            inner_dir+="_${param}"
        fi
    fi
}

run_base() {
    local dataset="$1"
    shift
    local algo="$1"
    shift
    local log_type="gd"
    local n_learners=1
    local optimizer="sgd"
    local extra_args_str=("$@")
    declare -A parameters

    # echo "${extra_args_str[@]}"
    echo "${extra_args_str[*]}"

    if [[ $dataset == synthetic* ]]; then
        extra_args_str+=("--input_dimension" "150" "--output_dimension" "2")
    fi
    while (("$#")); do
        # echo "$1"
        if [[ $1 == --* ]]; then
            local key=${1#--}
            shift
            if [[ $1 != --* && $1 != "" ]]; then
                parameters[$key]=$1
                shift
            else
                parameters[$key]=true
            fi
        else
            shift
        fi
    done
    parameters["sampling_rate"]=$sampling_rate

    if [[ -v parameters["fuzzy_m_scheduler"] ]]; then
        case ${parameters["fuzzy_m_scheduler"]} in
        "multi_step")
            parameters["fuzzy_m_scheduler"]="2step"
            ;;
        "cosine_annealing")
            parameters["fuzzy_m_scheduler"]="cosine"
            ;;
        esac
    fi
    local samp_dir=""
    local inner_dir=""
    case $algo in
    "FedAvg")
        set_inner_dir "locally_tune_clients"
        ;;
    "FuzzyFL")
        set_inner_dir "pre_rounds"
        set_inner_dir "fuzzy_m"
        if [[ ${parameters["fuzzy_m_scheduler"]} != "constant" ]]; then
            set_inner_dir "min_fuzzy_m"
        fi
        set_inner_dir "trans"
        set_inner_dir "fuzzy_m_scheduler"
        set_inner_dir "fuzzy_m_momentum"
        set_inner_dir "n_clusters"
        set_inner_dir "measurement"
        # samp_dir="/samp$sampling_rate"
        ;;
    "APFL")
        n_learners=2
        set_inner_dir "alpha"
        set_inner_dir "adaptive"
        ;;
    "clustered")
        sampling_rate=1
        ;;
    "AGFL")
        n_learners=4
        set_inner_dir "alpha"
        set_inner_dir "adaptive"
        set_inner_dir "pre_rounds"
        ;;
    "FedEM")
        n_learners=3
        set_inner_dir "m"
        ;;
    "FedProx")
        set_inner_dir "mu"
        optimizer="prox_sgd"
        ;;
    "pFedMe")
        set_inner_dir "mu"
        optimizer="prox_sgd"
        ;;
    "L2SGD")
        set_inner_dir "comm_prob"
        set_inner_dir "mu"
        optimizer="prox_sgd"
        ;;
    *) ;;
    esac

    # if [[ $algo != "FuzzyFL" ]];then
    #     set_inner_dir "sampling_rate"
    # fi
    # if [[ ${parameters[minibatch]} ]]; then
    #     log_type="batch"
    # fi
    # echo "${sampling_rate}"
    echo "$dataset"
    echo "${algo}"

    local log_dir="logs/$dataset/${algo}/lr_${lr}_samp_${sampling_rate}/${inner_dir}"
    # local save_path="chkpts/$dataset/${algo}${samp_dir}/${algo}_lr_${lr}${inner_dir}"
    # echo "$log_dir"
    if [ "${inner_dir}" != "" ]; then
        inner_dir="_${inner_dir}"
    fi
    local out_file="lr_${lr}${inner_dir}.log"

    check_dir "$log_dir"

    echo "$out_file"

    python run_experiment.py "$dataset" "$algo" --n_rounds $n_rounds --n_learners $n_learners --sampling_rate "$sampling_rate" --lr $lr \
        --lr_scheduler multi_step --optimizer $optimizer --bz $bz --local_steps $local_steps --log_freq $log_freq --seed 1234 --verbose 1 \
        "${extra_args_str[@]}"  --logs_root "$log_dir" >"$log_dir/$out_file"
}

run() {
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            run_base "$dataset" "$@" --minibatch
        done
    done
}


get_apfl_cmd(){
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            for alpha in "${alphas[@]}"; do
            commands+=("run_base $dataset  APFL --alpha $alpha --adaptive --minibatch")
            done
        done
    done
}

get_ordinary_cmd(){
    for dataset in "${DATA[@]}"; do
        for algo in "${algos[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            if [ "$algo" == "APFL" ]; then
            commands+=("run_base $dataset  APFL --alpha $alpha --adaptive --minibatch")
            else
                commands+=("run_base $dataset $algo --minibatch")
            fi
        done
        done
    done
}

get_prox_cmd(){
    for dataset in "${DATA[@]}"; do
    for algo in "${algos[@]}"; do
        for mu in "${mus[@]}"; do
            for sampling_rate in "${sampling_rates[@]}"; do
            if [ "$algo" == "L2SGD" ]; then
                commands+=("run_base $dataset $algo  --comm_prob ${comm_prob} --mu $mu --minibatch")
            else
                for comm_prob in "${comm_probs[@]}"; do
                    commands+=("run_base $dataset $algo --mu $mu --minibatch")
                done
            fi
        done
        done
    done
    done
}

get_fuzzy_cmd(){
    algo="FuzzyFL"
    for pre_rounds in "${pre_rounds_list[@]}"; do
        for fuzzy_m_scheduler in "${fuzzy_m_schedulers[@]}"; do
        for trans in "${trans_list[@]}"; do
        for fuzzy_m_momentum in "${fuzzy_m_momentums[@]}"; do
            for m in "${fuzzy_m_list[@]}"; do
            for measurement in "${measurements[@]}"; do
                if [ "$fuzzy_m_scheduler" == "cosine_annealing" ]; then
                    if [ "$(echo "$m > 1.7" | bc)" -eq 1 ]; then
                        min_m=$(echo "$m - 0.2" | bc)
                    else
                        min_m=$(echo "$m - 0.1" | bc)
                    fi
                fi
                # echo min_m="$min_m"
                for dataset in "${DATA[@]}"; do
                for sampling_rate in "${sampling_rates[@]}"; do
                    commands+=("run_base \"$dataset\" \"$algo\" --pre_rounds \"${pre_rounds}\" --fuzzy_m \"$m\"  --trans \"$trans\" --min_fuzzy_m \"${min_m}\" --fuzzy_m_scheduler \"${fuzzy_m_scheduler}\" --fuzzy_m_momentum \"${fuzzy_m_momentum}\" --measurement \"$measurement\" --minibatch")
                done
                done
            done
            done
            done
        done
        done
        done
}

run_fuzzy(){
    algo="FuzzyFL"
    for pre_rounds in "${pre_rounds_list[@]}"; do
        for fuzzy_m_scheduler in "${fuzzy_m_schedulers[@]}"; do
        for trans in "${trans_list[@]}"; do
        for fuzzy_m_momentum in "${fuzzy_m_momentums[@]}"; do
            for m in "${fuzzy_m_list[@]}"; do
            for min_m in "${min_m_list[@]}"; do
            for measurement in "${measurements[@]}"; do
                if [ "$fuzzy_m_scheduler" == "cosine_annealing" ]; then
                    if [ "$m" == "1.8" ] || [ "$m" == "2.0" ]; then
                    min_m=$(python -c "print($m - 0.2)")
                    fi
                    min_m=$(python -c "print($m - 0.1)")
                fi
                for dataset in "${DATA[@]}"; do
                for sampling_rate in "${sampling_rates[@]}"; do
                    run_base "$dataset" "$algo" --pre_rounds "${pre_rounds}" --fuzzy_m "$m" --min_fuzzy_m "${min_m}" --trans "$trans" --fuzzy_m_scheduler "${fuzzy_m_scheduler}" --fuzzy_m_momentum "${fuzzy_m_momentum}" --measurement "$measurement" --minibatch
                done
                done
            done
            done
            done
        done
        done
        done
    done
}

run_avg_adap() {
    algo="FedAvg"
    run   $algo --locally_tune_clients
}
run_avg() {
    algo="FedAvg"
    run $algo
}
run_apfl() {
    algo="APFL"
    for alpha in "${alphas[@]}"; do
        run  $algo --alpha "$alpha" --adaptive
    done
}

run_agfl() {
    algo="AGFL"
    for alpha in "${alphas[@]}"; do
        run  $algo --alpha "$alpha" --adaptive
    done
}

run_local() {
    algo="local"
    run $algo
}

run_em() {
    algo="FedEM"
    run  $algo
}

run_clustered() {
    algo="clustered"
    run  $algo
}

run_avgem() {
    algos=("FedAvg" "FedEM")
    for algo in "${algos[@]}"; do
        run  $algo
    done
}

run_l2gd() {
    algo="L2SGD"
    for comm_prob in "${comm_probs[@]}"; do
        for mu in "${mus[@]}"; do
            run  $algo --comm_prob "$comm_prob" --mu "$mu"
        done
    done
}

run_pfedme() {
    algo="pFedMe"
    for mu in "${mus[@]}"; do
        run   $algo --mu "$mu"
    done
}

run_prox() {
    algo="FedProx"
    for mu in "${mus[@]}"; do
        run  $algo --mu "$mu"
    done
}

check_dir() {
    local dir_path="$1"

    if [ -d "$dir_path" ]; then
        echo -e "Directory $dir_path exists. \nIt will be deleted in 10 seconds unless you cancel the operation. \
        \nDo you want to remove it? (Yy/n)"

        count=10
        while [ $count -gt 0 ]; do
            echo -ne "\rTime remaining: $count s..."
            if read -r -t 1 user_input; then
                if [[ "$user_input" =~ ^(y|Y)$ ]]; then
                    rm -rf "$dir_path"
                    echo -e "\nDirectory $dir_path removed."
                    break

                elif [ "$user_input" = "n" ]; then
                    echo -e "\nDirectory $dir_path not removed. Now will rename $dir_path"

                    suffix=1
                    while [ -d "${dir_path}_$suffix" ]; do
                        suffix=$((suffix + 1))
                    done
                    new_dir_path="${dir_path}_$suffix"

                    mv "$dir_path" "$new_dir_path"
                    echo "Directory $dir_path renamed to $new_dir_path"
                    break
                fi
            fi
            count=$((count - 1))
        done
        if [ $count -eq 0 ]; then
            echo -e "\nDirectory $dir_path removed due to no response."
            rm -rf "$dir_path"
        fi
    fi
    mkdir -p "$dir_path"
    echo "Directory $dir_path created successfully"
}

run_gd() {
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            run_base "$dataset" "$@"
        done
    done
}

show_dict() {
    local dict="$1"
    for key in "${!dict[@]}"; do
        value="${dict[$key]}"
        echo "$key: $value"
    done
}
modify_dict_key() {
    local dict_var="$1"
    local old_key="$2"
    local new_key="$3"

    eval "${dict_var}[\"$new_key\"]=\"\${${dict_var}[\"$old_key\"]}\""

    unset "${dict_var}[$old_key]"
}

string_in_array(){
    my_list=$1
    search_string=$2
    found=false
    for item in "${my_list[@]}"
    do
    if [ "$item" == "$search_string" ]
    then
        found=true
        break
    fi
    done
}