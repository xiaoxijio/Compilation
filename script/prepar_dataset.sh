#!/bin/bash

function prepar_ILSVRC2012_dataset
{
    if [[ ! -d "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012/ILSVRC2012_img_val/" ]]; then
        echo "prepar ILSVRC2012 dataset..."
        fetch_evaluation_datasets "ILSVRC2012/ILSVRC2012_img_val.tar" "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset"
        mkdir -p ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012/ILSVRC2012_img_val
        tar -xvf "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012_img_val.tar" -C "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/ILSVRC2012/ILSVRC2012_img_val/"
    else
        echo "check ILSVRC2012 dataset is ready"
    fi
}

function prepar_COCO2017_dataset
{
    if [[ ! -d "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/val2017" ]]; then
        echo "prepar COCO2017 dataset..."
        fetch_evaluation_datasets "COCO2017/val2017.zip" "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/"
        fetch_evaluation_datasets "COCO2017/annotations/annotations_trainval2017.zip" "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/"
        mkdir -p ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/
        unzip ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/val2017.zip -d ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/
        unzip ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/annotations_trainval2017.zip -d ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/COCO2017/
    else
        echo "check ILSVRC2012 dataset is ready"
    fi
}

function prepar_nuscenes_dataset
{
    sample_type=$1
    enable_pub=$2
    real_data_path="${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/nuscenes"
    # 下载并解压数据集
    if [[ ! -d "${real_data_path}" ]]; then
        echo "prepar nuscenes dataset..."
        mkdir -p "${real_data_path}"
        fetch_evaluation_datasets "nuscenes/*" "${real_data_path}/"
        tar -zxvf "${real_data_path}/bevdetv3-nuscenes_gt_database.tar.gz" -C "${real_data_path}/"
        tar -zxvf "${real_data_path}/maps.tar.gz" -C "${real_data_path}/"
        tar -zxvf "${real_data_path}/samples.tar.gz" -C "${real_data_path}/"
        tar -zxvf "${real_data_path}/v1.0-trainval.tar.gz" -C "${real_data_path}/"
        tar -zxvf "${real_data_path}/gts.tar.gz" -C "${real_data_path}/"
        tar -zxvf "${real_data_path}/sweeps.tar.gz" -C "${real_data_path}/"
        unzip "${real_data_path}/can_bus.zip" -d "${real_data_path}/"
    else
        echo "check nuscenes dataset downloade ready"
    fi
    if [[ ${sample_type} = "bevformer" ]]; then
        nuscenes_data_root_path="${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate/data/"
        # 软连接数据集到目标路径
        if [[ ! -d "${nuscenes_data_root_path}/nuscenes" ]]; then
            mkdir -p $nuscenes_data_root_path
            ln -s $real_data_path $nuscenes_data_root_path
        fi
        # 数据集预处理
        if [[ ! -d "${nuscenes_data_root_path}/../preprocessed_imgs_bevformer/" ]]; then
            cd "${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate"
            python save_preprocessed_imgs.py --config bevformer --save_path ./preprocessed_imgs_bevformer/
            cd -
        else
            echo "check bevformer nuscenes dataset prepar ready"
        fi
    elif [[ ${sample_type} = "fastbev++" ]]; then
        nuscenes_data_root_path="${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate/data/"
        if [[ ! -d "${nuscenes_data_root_path}/nuscenes" ]]; then
            mkdir -p $nuscenes_data_root_path
            ln -s $real_data_path $nuscenes_data_root_path
        else
            echo "check nuscenes dataset prepar ready"
        fi
    elif [[ ${sample_type} = "bevdet" ]]; then
        nuscenes_data_root_path="${NPU_SAMPLE_ROOT_DIR}/app/vision_bev/mod/common/evaluate/data/"
        if [[ ! -d "${nuscenes_data_root_path}/nuscenes" ]]; then
            mkdir -p $nuscenes_data_root_path
            ln -s $real_data_path $nuscenes_data_root_path
        else
            echo "check nuscenes dataset prepar ready"
        fi
    elif [[ ${sample_type} = "flashocc" ]]; then
        nuscenes_data_root_path="${NPU_SAMPLE_ROOT_DIR}/app/vision_occ/evaluate/data"
        if [[ ! -d "${nuscenes_data_root_path}/nuscenes" ]]; then
            mkdir -p $nuscenes_data_root_path
            ln -s $real_data_path $nuscenes_data_root_path
        else
            echo "check nuscenes dataset prepar ready"
        fi
    elif [[ ${sample_type} = "lss" ]]; then
        nuscenes_data_root_path="${NPU_SAMPLE_ROOT_DIR}/app/vision_lss/lss/data"
        if [[ ! -d "${nuscenes_data_root_path}/nuscenes" ]]; then
            mkdir -p $nuscenes_data_root_path
            ln -s $real_data_path $nuscenes_data_root_path
        else
            echo "check nuscenes dataset prepar ready"
        fi
    else
        echo "please check param input!"
    fi

}

function prepar_GOT10K_dataset
{
    if [[ ! -d "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/GOT_10K/" ]]; then
        echo "prepar GOT_10K dataset..."
        fetch_evaluation_datasets "mixformer/GOT_10K.tar.gz" "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset"
        mkdir -p ${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/GOT_10K
        tar -xzvf "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/GOT_10K.tar.gz" -C "${NPU_SAMPLE_ROOT_DIR}/data/evaluate_dataset/"
    else
        echo "check GOT_10K dataset is ready"
    fi
}