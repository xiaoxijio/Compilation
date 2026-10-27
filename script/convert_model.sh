set -eo pipefail
CUR_DIR=$(
    cd $(dirname $0)
    pwd
)
# 配置参数
MAX_CONCURRENT=32            # 最大并发数
LOG_DIR="${CUR_DIR}/../logs" # 日志目录
FAILED_TASKS=()              # 失败任务记录
mkdir -p "$LOG_DIR"
board_steps=(
    "00_init.sh"
    "01_check.sh"
    "02_preprocess.sh"
    "03_omg.sh"
)
sim_steps=(
    "00_init.sh"
    "01_check.sh"
    "02_preprocess.sh"
    "03_omg.sh"
    "04_build.sh"
    "05_inference.sh"
)
board_testcases=(
    "api_tutorial/binary_input"
    "api_tutorial/yuv420sp_input"
    "api_tutorial/custom_abs"
    "app/vision_2d/classification/mobilenetv2"
    "app/vision_2d/classification/resnet50"
    "app/vision_2d/classification/resnet152"
    "app/vision_2d/mod/yolov5"
    "app/vision_2d/mod/yolov7"
    "app/vision_2d/mod/yolov10"
    "app/vision_2d/mod/yolov8"
    "app/vision_bev/mod/bevdet"
    "app/vision_bev/mod/fastbev++"
    "app/stereo/lightstereo"
    "app/vision_occ/flashocc"
    "app/vision_lss/lss"
    "app/mixformer/mixformer"
)
sim_testcases=(
    "api_tutorial/binary_input"
    "api_tutorial/yuv420sp_input"
    "app/vision_2d/classification/mobilenetv2"
)

# 并发控制初始化
fifo="/tmp/$$.fifo"
mkfifo "$fifo"
exec 5<>"$fifo" # 使用文件描述符5
rm -f "$fifo"
for ((i = 0; i < MAX_CONCURRENT; i++)); do
    echo >&5 # 初始化令牌
done

function convert_and_run() {
    local testcase=$1
    local target=$2
    local curr_env=$3
    local model_name=${testcase##*/}
    local log_file="${LOG_DIR}/${model_name}_${target}.log"
    local status_file="${LOG_DIR}/${model_name}_${target}.status"
    if [ "X$curr_env" = "Xsim" ]; then
        local steps=("${sim_steps[@]}")
    else
        local steps=("${board_steps[@]}")
    fi
    # 在子shell中执行避免污染父进程环境
    (
        # 使用绝对路径并锁定工作目录
        local work_dir="${CUR_DIR}/../${testcase}"
        cd "$work_dir" || {
            echo "[FATAL] Cannot cd to $work_dir" | tee -a "$log_file"
            exit 2
        }

        # 记录开始信息
        echo "[${model_name}]===== TASK START: $(date '+%Y-%m-%d %T') =====" | tee -a "$log_file"
        echo "[${model_name}]Work Directory: $(pwd)" | tee -a "$log_file"
        echo "[${model_name}]Target Platform: $target" | tee -a "$log_file"

        # 执行所有步骤
        for step in "${steps[@]}"; do
            local step_start=$(date +%s)
            echo "[${model_name}]---- [STEP BEGIN] $step ($(date '+%H:%M:%S')) ----" | tee -a "$log_file"

            # 带超时和错误捕获的执行
            bash -eo pipefail "./$step" >>"$log_file" 2>&1
            local exit_code=$?
            if [[ ${exit_code} -ne 0 ]]; then
                echo "[${model_name}]---- [STEP FAILED] $step (exit code:${exit_code}) ----" | tee -a "$log_file"
                echo "[${model_name}]failed_step=$step" | tee "$status_file"
                echo "[${model_name}]exit_code=${exit_code}" | tee -a "$status_file"
                exit 1
            fi
            local duration=$(($(date +%s) - step_start))
            echo "[${model_name}]---- [STEP PASSED] $step (duration: ${duration}s) ----" | tee -a "$log_file"
        done

        echo "[${model_name}]===== TASK COMPLETED SUCCESSFULLY =====" | tee -a "$log_file"
        echo "[${model_name}]success" | tee "$status_file"
    )
    return $?
}

function batch_convert_model() {
    local prev_dir="$(pwd)"
    local temp_file=$(mktemp)
    local target=$1
    local curr_env=$2
    export NPU_TOOLCHAIN_PATH="${prev_dir}/npu_toolchain"
    if [ "X$curr_env" = "Xsim" ]; then
        local testcases=("${sim_testcases[@]}")
        sed -i "s/\"board\"/\"simulator\"/g" ${CUR_DIR}/platform_config.sh
    else
        local testcases=("${board_testcases[@]}")
        sed -i "s/\"simulator\"/\"board\"/g" ${CUR_DIR}/platform_config.sh
        if [ "X$target" = "XLQ560PlusV100" ]; then
            testcases=("${testcases[@]:0:${#testcases[@]}-3}")
        fi
    fi

    sed -i '0,/TARGET="LQ560.*"/ s//TARGET="'"${target}"'"/'  ${CUR_DIR}/platform_config.sh
    FAILED_TASKS=()
    for testcase in ${testcases[@]}; do
        read -u5 # 获取令牌
        {
            if ! convert_and_run $testcase $target $curr_env; then
                echo "$testcase" >>"$temp_file" # 将失败用例写入临时文件
            fi
            echo >&5 # 释放令牌
        } &
    done

    # 等待所有后台任务完成
    wait
    exec 5>&- # 关闭管道

    # 结果汇总
    # 从临时文件读取失败任务到数组
    if [[ -s "$temp_file" ]]; then
        mapfile -t FAILED_TASKS <"$temp_file"
    fi
    echo "===== Execution Summary ====="
    echo "Total tasks: ${#testcases[@]}"
    echo "Successful: $((${#testcases[@]} - ${#FAILED_TASKS[@]}))"
    if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
        echo "Failed tasks (${#FAILED_TASKS[@]}):"
        printf ' - %s\n' "${FAILED_TASKS[@]}"
        for testcase in ${FAILED_TASKS[@]}; do
            local model_name=${testcase##*/}
            local log_file="${LOG_DIR}/${model_name}_${target}.log"
            local status_file="${LOG_DIR}/${model_name}_${target}.status"
            echo -e "\033[33m\n\n-----------------ERR INFO-----------------------\n\n\033[0m"
            cat $status_file
            cat $log_file
            echo -e "\033[33m\n\n-----------------END-----------------------\n\n\033[0m"

        done
        return 1 # 若有失败任务则返回非零状态码
    else
        echo "All tasks completed successfully"
        rm -rf ${LOG_DIR}
    fi
    sed -i "s/\"board\"/\"simulator\"/g" ${CUR_DIR}/platform_config.sh

}
function convert_model_board_v100() {
    batch_convert_model LQ560PlusV100 board
}

function convert_model_board_v200() {
    batch_convert_model LQ560PlusV200 board
}
function convert_model_v100() {
    batch_convert_model LQ560PlusV100 sim
}

function convert_model_v200() {
    batch_convert_model LQ560PlusV200 sim
}
function parse_args() {
    for arg in $@; do
        if [ "X$arg" = "Xconvert_model_board_v100" ]; then
            convert_model_board_v100
            exit 0
        elif [ "X$arg" = "Xconvert_model_board_v200" ]; then
            convert_model_board_v200
            exit 0
        elif [ "X$arg" = "Xconvert_model_v100" ]; then
            convert_model_v100
            exit 0
        elif [ "X$arg" = "Xconvert_model_v200" ]; then
            convert_model_v200
            exit 0
        fi
    done
}
function main() {
    parse_args $@
}

main $@
