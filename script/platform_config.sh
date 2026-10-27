# config.sh
TARGET="LQ560ProV200"        # [LQ560PlusV100/LQ560MiniV110/LQ560PlusV200/LQ560ProV200.etc]
PLATFORM="board" # [simulator/board]
BOARD_IP="10.16.81.27"
PORT="1234"
RED='\033[0;31m'
NC='\033[0m'
ENABLE_OPENSOURCE_ONNX=0 # 选择开源/魔改 ONNX, 1:开源，0: 魔改

if [ -n "$NPU_TOOLCHAIN_PATH" ]; then
    OMG="${NPU_TOOLCHAIN_PATH}/linux/omg/bin/omg"
fi
if [ ! -f "$OMG" ]; then
    NPU_TOOLCHAIN_PATH=${NPU_SAMPLE_ROOT_DIR}/../..
    OMG="${NPU_TOOLCHAIN_PATH}/omg_toolchain/omg/bin/omg"
    if [ ! -f "$OMG" ]; then
        OMG="${NPU_TOOLCHAIN_PATH}/linux/omg/bin/omg"
        if [ ! -f "$OMG" ]; then
            OMG="/avp/npu/omg/bin/omg"
            if [ ! -f "$OMG" ]; then
                NPU_TOOLCHAIN_PATH="/avprelease/dailybuild/$(date +%Y-%m-%d)/release_unzip/npu_toolchain" # npu_toolchain发布包根目录
                OMG="${NPU_TOOLCHAIN_PATH}/linux/omg/bin/omg"
            else
                echo "OMG PATH ERROR:${OMG}"
            fi
        fi
    fi
fi

function check_file_dir_exist() {
    if [ -e "${1}" ]; then
        echo "check ${1} ok"
        return 0
    else
        echo -e  "${RED}${1} is not ready${NC}"
        exit 1
    fi
}

function print_and_run_cmd
{
    local PS4='Running cmd: '
    local -
    set -o xtrace

    "$@"
}

function check_target
{
    target="${1}"
    if [[ "$target" = "LQ560PlusV100" || "$target" = "LQ560MiniV100" ||  \
         "$target" = "LQ560LiteV100" || "$target" = "LQ560TinyV100" ||  \
         "$target" = "SD3589C"       || "$target" = "SD3589D" ]]; then
        echo "0"
        return
    fi
    echo "1"
}

function fetch_evaluation_datasets
{
    # 检查参数数量是否正确
    if [ $# -ne 2 ]; then
        echo "usage: fetch_evaluation_datasets <数据集相对路径> <数据集保存路径>"
        echo "   eg: fetch_evaluation_datasets "ILSVRC2012/ILSVRC2012_img_val.tar" "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset""
        return 1
    fi

    relative_path="$1"
    save_path="$2"

    echo "fetch_evaluation_datasets ${relative_path} ${save_path}"

    mkdir -p "$(dirname "$save_path")" || {
        echo "can not mkdir -p $save_path"
        return 1
    }

    wget -c \
         --ftp-user=ftpuser1 \
         --ftp-password="Z3j+7a&DQ*" \
         ftp://extftp.oritek.com.cn/ai_datasets/$relative_path \
         -P $save_path && {
        echo "SUCCESS: fetch '$relative_path' to '$save_path'"
        return 0
    }

    echo "FAIL: fetch '$relative_path' fail."
    return 1
}
