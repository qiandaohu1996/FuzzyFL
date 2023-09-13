#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC2016

declare -g DATA=()
declare -g algos=()

# algorithm parameters
declare -g algo=""
declare -g bz=256
declare -g local_steps=1
declare -g max_concurrent_processes=1

declare -g n_rounds=200
declare -g log_freq=1

# for all
declare -g seed=2222

declare -g n_learners=1

declare -g sampling_rates=("0.5")
declare -g learner_rates=()
# for fuzzyFL
declare -g pre_rounds_list=("50")
declare -g fuzzy_m_schedulers=("constant")
declare -g fuzzy_m_list=("1.8")
declare -g min_m_list=("1.5")
declare -g min_m="1.5"
declare -g trans_list=("0.75")
declare -g fuzzy_m_momentums=("0.8")
declare -g measurements=("euclid")
declare -g n_clusters=3
declare -g top=3
# for fedsoft
declare -g tau=4

# for l2sgd fedprox and pfedMe fedsoft
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
        ["n_clusters"]="_clusters_%s"
        ["n_learners"]="learners_%s"
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

run() {
    local dataset="$1"
    shift
    local algo="$1"
    shift
    local log_type="gd"
    local optimizer="sgd"
    local extra_args_str=("$@")
    declare -A parameters
    # echo "${extra_args_str[@]}"
    # echo "${extra_args_str[*]}"
    if [[ $dataset == synthetic* ]]; then
        extra_args_str+=("--input_dimension" "150" "--output_dimension" "2")
    fi
    local index=0
    while [ $index -lt ${#extra_args_str[@]} ]; do
        # echo "${extra_args_str[$index]}"s
        if [[ ${extra_args_str[$index]} == --* ]]; then
            local key=${extra_args_str[$index]#--}
            index=$((index+1))
            if [[ ${extra_args_str[$index]} != --* && ${extra_args_str[$index]} != "" ]]; then
                parameters[$key]=${extra_args_str[$index]}
                index=$((index+1))
            else
                parameters[$key]=true
            fi
        else
            index=$((index+1))
        fi
    done
    # parameters["sampling_rate"]=$sampling_rate
    # show_dict parameters
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
    sampling_rate=${parameters["sampling_rate"]}
    # seed=${parameters["seed"]}
    lr=${parameters["lr"]}
    echo "${sampling_rate}"
    if [[ $algo == "FedEM" ]]; then
        parameters+=(["n_learners"]=${n_learners})
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
        if [[ ${n_clusters} -gt 3 ]];then
            set_inner_dir "n_clusters"
            set_inner_dir "top"
        fi
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
        set_inner_dir "m"
        set_inner_dir "n_learners"
        ;;
    "FedProx")
        set_inner_dir "mu"
        optimizer="prox_sgd"
        ;;
    "pFedMe")
        set_inner_dir "mu"
        optimizer="prox_sgd"
        ;;
    "FedSoft")
        set_inner_dir "mu"
        set_inner_dir "n_clusters"
        optimizer="soft_proxsgd"
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
    echo "$dataset"
    echo "$algo"
    echo "${inner_dir}"
    first_dir="lr_${lr}_samp_${sampling_rate}_seed_${seed}"
    # first_dir="lr_${lr}_samp_${sampling_rate}_seed_$seed"
    local log_dir="logs/$dataset/${algo}/${first_dir}/${inner_dir}"
    # local save_path="chkpts/$dataset/${algo}${samp_dir}/${algo}_lr_${lr}${inner_dir}"
    # echo "$log_dir"
    if [ "${inner_dir}" != "" ]; then
        inner_dir="_${inner_dir}"
    fi
    local out_file="lr_${lr}${inner_dir}.log"

    check_dir "$log_dir"

    echo "$out_file"
    # echo "$dataset" "$algo" "${extra_args_str[@]}"  --n_rounds $n_rounds --n_learners $n_learners --lr $lr \
    #     --lr_scheduler multi_step --optimizer $optimizer --bz $bz --local_steps $local_steps --log_freq $log_freq --seed 1234 --verbose 1 \
    #      --logs_root "$log_dir" >"$log_dir/$out_file"
    python run_experiment.py "$dataset" "$algo" "${extra_args_str[@]}"  --n_rounds ${n_rounds} --n_learners ${n_learners} --lr_scheduler "multi_step" --optimizer $optimizer --bz $bz --local_steps ${local_steps} --log_freq ${log_freq} --seed $seed --verbose 1 \
         --logs_root "${log_dir}" >"$log_dir/${out_file}"
}



# get_apfl_cmd(){
#     for dataset in "${DATA[@]}"; do
#         for sampling_rate in "${sampling_rates[@]}"; do
#             for alpha in "${alphas[@]}"; do
#             commands+=("run $dataset --sampling_rate ${sampling_rate}  APFL --alpha $alpha --adaptive --minibatch")
#             done
#         done
#     done
# }

get_ordinary_cmd(){
    for algo in "${algos[@]}"; do
        for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
        for lr in "${learner_rates[@]}"; do
            if [ "$algo" == "APFL" ]; then
            commands+=("run $dataset APFL --sampling_rate ${sampling_rate} --alpha $alpha --adaptive --minibatch")
            else
            commands+=("run $dataset $algo --lr $lr --sampling_rate ${sampling_rate} --minibatch")
            fi
        done
        done
        done
    done
}

get_soft_cmd(){
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
        for lr in "${learner_rates[@]}"; do
            for mu in "${mus[@]}"; do
                commands+=("run $dataset FedSoft --lr $lr --sampling_rate ${sampling_rate} --n_clusters ${n_clusters} --mu $mu --tau $tau --minibatch")
            done
        done
        done
    done
}

get_prox_cmd(){
    for algo in "${algos[@]}"; do
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
        for mu in "${mus[@]}"; do
                for lr in "${learner_rates[@]}"; do
            if [ "$algo" == "L2SGD" ]; then
                commands+=("run $dataset $algo --lr $lr --sampling_rate ${sampling_rate} --comm_prob ${comm_prob} --mu $mu --minibatch")
            else
                for comm_prob in "${comm_probs[@]}"; do
                    commands+=("run $dataset $algo --lr $lr --sampling_rate ${sampling_rate}  --mu $mu --minibatch")
                done
            fi
        done
        done
        done
    done
    done
}

get_fuzzy_cmd(){
    algo="FuzzyFL" 
    for dataset in "${DATA[@]}"; do
    for lr in "${learner_rates[@]}"; do
    for sampling_rate in "${sampling_rates[@]}"; do
        for pre_rounds in "${pre_rounds_list[@]}"; do
        for m in "${fuzzy_m_list[@]}"; do
        for fuzzy_m_momentum in "${fuzzy_m_momentums[@]}"; do
        for fuzzy_m_scheduler in "${fuzzy_m_schedulers[@]}"; do
            for trans in "${trans_list[@]}"; do
            for measurement in "${measurements[@]}"; do
                if [ "$fuzzy_m_scheduler" == "cosine_annealing" ]; then
                    if [ "$(python -c "print(1 if $m < 1.8 else 0)")" -eq 1 ]; then
                        min_m=$(calculate_min_m 0.1)
                    elif [ "$(python -c "print(1 if $m < 2.3 else 0)")" -eq 1 ]; then
                        min_m=$(calculate_min_m 0.2)
                    else
                        min_m=$(calculate_min_m 0.3)
                    fi
                    # if [ "$(echo "$m < 1.8" | bc)" -eq 1 ]; then
                    #     min_m=$(echo "$m - 0.1" | bc)
                    # elif [ "$(echo "$m < 2.3" | bc)" -eq 1 ]; then
                    #     min_m=$(echo "$m - 0.2" | bc)
                    # else 
                    #     min_m=$(echo "$m - 0.3" | bc)
                    # fi
                fi
                
                # echo min_m="$min_m"
                    commands+=("run $dataset $algo --lr $lr --sampling_rate ${sampling_rate} --pre_rounds ${pre_rounds} --n_clusters ${n_clusters} --top $top  --fuzzy_m $m  --trans $trans --min_fuzzy_m ${min_m} --fuzzy_m_scheduler ${fuzzy_m_scheduler} --fuzzy_m_momentum ${fuzzy_m_momentum} --measurement $measurement --minibatch")
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

calculate_min_m() {
    value_to_subtract="$1"
    python -c "result = $m - $value_to_subtract; print(int(result) if result == int(result) else round(result, 1))"
}

# read -p "Enter the value of m: " m



# echo "Value of min_m: $min_m"

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
                    run "$dataset" "$algo" --sampling_rate "${sampling_rate}" --pre_rounds "${pre_rounds}" --fuzzy_m "$m" --min_fuzzy_m "${min_m}" --trans "$trans" --fuzzy_m_scheduler "${fuzzy_m_scheduler}" --fuzzy_m_momentum "${fuzzy_m_momentum}" --measurement "$measurement" --minibatch
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

run_base() {
    algo=$1
    shift
    for dataset in "${DATA[@]}"; do
    for sampling_rate in "${sampling_rates[@]}"; do
        run "$dataset" "$algo"  --sampling_rate "${sampling_rate}" --minibatch "${@}" 
    done 
    done
}

run_local() {
    algo="local"
    run_base $algo
}

run_em() {
    algo="FedEM"
    run_base  $algo 
}

run_clustered() {
    algo="clustered"
    run_base  $algo
}

run_avgem() {
    algos=("FedAvg" "FedEM")
    for algo in "${algos[@]}"; do
        run_base  "$algo"
    done
}

run_l2gd() {
    algo="L2SGD"
    for comm_prob in "${comm_probs[@]}"; do
        for mu in "${mus[@]}"; do
            run_base  "$algo" --comm_prob "$comm_prob" --mu "$mu"
        done
    done
}

run_pfedme() {
    algo="pFedMe"
    for mu in "${mus[@]}"; do
        run_base   "$algo" --mu "$mu"
    done
}

run_prox() {
    algo="FedProx"
    for mu in "${mus[@]}"; do
        run_base  "$algo" --mu "$mu"
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

                elif [[ "$user_input" =~ ^(N|n)$ ]]; then
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
            run "$dataset" "$@"
        done
    done
}
print_programs(){
echo commands length: ${#commands[@]}
for program in "${commands[@]}"; do
    echo "$program"
done
echo 
}

multi_run() { 
    current_processes=0
    for cmd in "${commands[@]}"; do
        while [ $current_processes -ge $max_concurrent_processes ]; do
            wait -n  # 等待任何子进程结束
            ((current_processes--))  # 子进程结束，计数减1
            echo current_processes = "${current_processes}"
        done
        eval "$cmd" &
        echo 
        # echo "$cmd" 
        echo process "$!" start
        ((current_processes++))
        echo current_processes = "${current_processes}"
        sleep 10
    done

    # 等待剩余的子进程结束
    while [ $current_processes -gt 0 ]; do
        wait -n
        ((current_processes--))
        echo current_processes = "${current_processes}"
    done

    echo "All programs have finished execution."
commands=()

}

show_dict() {
    local -n dict="$1"
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