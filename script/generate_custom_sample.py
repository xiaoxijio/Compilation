# python python generate_custom_sample.py -m [onnx路径] -i [input_cfg.yml路径]
# 按照给定的onxx，生成对应的sample工程
# 如果给定yml文件则直接使用该yml
# 如果未给定yml文件夹，根据onnx生成一个input_cfg.yml文件，其中calib_path，calib_type，input_type，preprocess_input_type，preprocess_output_type都需要手动配置

import os
import sys
import shutil
import argparse
import yaml
import onnx
import onnxruntime
import ruamel.yaml

model_reference_cfg = '''
quant:
  dataset:
    calib_path: # If set null, LLM dataset is built in and CNN dataset is set 0
    quant_batch_size: 1
  acc_dump:
    enable_dump: false
    dump_path: ./result.json
  activation:
    algorithm: MaxMin
    symmetric: false
    quant_type: "S8"
  weight:
    algorithm: MaxMin
    quant_type: "S8"
    per_channel: false
executor:
  cpu_settings:
    thread_num: 4
  gpu_settings:
    dev_id: 0
inputs:
  - input_name: data
    calib_type: # [Image/Binary]
    input_type: # [Image/Binary]
    reverse_rgb_order: false
    data_layout: NCHW
    shape: [1, 3, 224, 224]
    preprocess:
      resize:
        letter_box: false
        height: 256
        width: 256
      crop:
        method: absolute
        height: 224
        width: 224
      dtc:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
'''

model_input_cfg = '''
    input_name: data
    calib_type: # [Image/Binary]
    input_type: # [Image/Binary]
    reverse_rgb_order: false
    data_layout: NCHW
    shape: [1, 3, 224, 224]
    preprocess:
        resize:
            letter_box: false
            height: 256
            width: 256
        crop:
            method: absolute
            height: 224
            width: 224
        dtc:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
'''

def set_input_info(input, input_info):
    input["input_name"] = input_info["name"]
    input["shape"] = input_info["shape"]
    input["preprocess"]["resize"]["height"] = input_info["resize"][0]
    input["preprocess"]["resize"]["width"]  = input_info["resize"][1]
    input["preprocess"]["crop"]["height"]   = input_info["crop"][0]
    input["preprocess"]["crop"]["width"]    = input_info["crop"][1]

def remove_initializer_from_input(model_path, tmp_model_path):
    model = onnx.load(model_path)

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, tmp_model_path)

def get_model_input_info(model_name, resize_size = [256, 256]):
    tmp_model_path = "tmp.onnx"
    tmp_model_path = model_name
    remove_initializer_from_input(model_name, tmp_model_path)

    provider = "CPUExecutionProvider"
    onnx_session = onnxruntime.InferenceSession(tmp_model_path, providers=[provider])

    input_tensors = onnx_session.get_inputs()
    inputs_info = []
    for input_tensor in input_tensors:
        shape = input_tensor.shape
        print ("input shape:", shape)
        for i in range(len(shape)):
            if str(shape[i]).isdigit() is False:
                print ("change input shape %d from %r to %d" % (i, shape[i], 1))
                shape[i] = 1

        # update resize_size
        if resize_size[0] < shape[-2]:
            resize_size[0] = shape[-2]
        if resize_size[1] < shape[-1]:
            resize_size[1] = shape[-1]

        inputs_info.append({
            "name" : input_tensor.name,
            "type" : input_tensor.type,
            "shape": shape,
            "resize" : resize_size,
            "crop"   : [shape[-2], shape[-1]] # chw
        })

    print(inputs_info)
    return inputs_info

def gen_model_input_cfg(inputs_info, target_dir):
    yaml = ruamel.yaml.YAML()
    config = yaml.load(model_reference_cfg)

def gen_model_input_cfg(inputs_info, target_dir):
    import ruamel.yaml  # 确保导入
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    config = yaml.load(model_reference_cfg)

    for i in range(len(inputs_info)):
        input_info = inputs_info[i]
        if i == 0:
            input = config["inputs"][0]
        else:
            input = yaml.load(model_input_cfg)
            config["inputs"].append(input)

        set_input_info(input, input_info)

        # 设置 input_type 为 None，并添加注释
        input['input_type'] = None
        input.yaml_add_eol_comment("请指定输入数据格式: Image/Binary", key='input_type')

        preprocess = input.get('preprocess', {})
        if not isinstance(preprocess, ruamel.yaml.comments.CommentedMap):
            preprocess = ruamel.yaml.comments.CommentedMap()

        # 设置 preprocess_input_type
        preprocess['preprocess_input_type'] = None
        preprocess.yaml_add_eol_comment("可选项: RGBPlanar/YUV420SP/YVU420SP/YUV422SP/YUV400/RAW_RGGB/RAW_BGGR/RAW_GBRG/RAW_GRBG", key='preprocess_input_type')

        # 设置 preprocess_output_type
        preprocess['preprocess_output_type'] = None
        preprocess.yaml_add_eol_comment("可选项: RGBPlanar/Gray/YUV444Planar/ BGGRPlanar/GBRGPlanar/GRBGPlanar/RGGBPlanar", key='preprocess_output_type')

        input['preprocess'] = preprocess

    if 'quant' in config and 'dataset' in config['quant']:
        dataset = config['quant']['dataset']
        if 'calib_path' in dataset and dataset['calib_path'] is None:
            del dataset['calib_path']
        dataset.insert(0, 'calib_path', None)
        dataset.yaml_add_eol_comment("请根据实际情况填写量化校准集路径", key='calib_path')

    for input in config.get('inputs', []):
        if 'calib_type' in input and input['calib_type'] is None:
            del input['calib_type']
        input.insert(1, 'calib_type', None)
        input.yaml_add_eol_comment("请根据实际情况填写量化时校准集的格式Image/Binary", key='calib_type')

    with open(target_dir + "/input_cfg.yml", "w") as f:
        yaml.dump(config, f)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.dirname(SCRIPT_DIR)  # sample/

CUSTOM_DIR = os.path.join(SAMPLE_DIR, "app", "custom")
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")
TOP_CMAKE_FILE = os.path.join(SAMPLE_DIR, "CMakeLists.txt")

def get_model_name(model_path):
    return os.path.splitext(os.path.basename(model_path))[0]

def ensure_directory(force, target_dir):
    if os.path.exists(target_dir):
        if force:
            print(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            print(f"Error: Directory '{target_dir}' already exists.")
            print("Use -f to force overwrite.")
            sys.exit(1)
    os.makedirs(target_dir)
    print(f"Created directory: {target_dir}")


def update_cmake_file(model_name, top_cmake_file):
    subdir_line = f"add_subdirectory(./app/custom/{model_name})\n"

    if not os.path.exists(top_cmake_file):
        print(f"Error: Top-level CMakeLists.txt does not exist at '{top_cmake_file}'")
        print("This script requires an existing top-level CMakeLists.txt to integrate the model.")
        sys.exit(1)

    with open(top_cmake_file, "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if line.strip() != subdir_line.strip()]

    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append(subdir_line)

    with open(top_cmake_file, "w") as f:
        f.writelines(lines)

    print(f"Updated CMakeLists.txt with '{subdir_line.strip()}'")


def replace_model_name_and_copy(src_path, dst_path, model_name):
    with open(src_path, "r") as f:
        content = f.read()

    content = content.replace("{{MODEL_NAME}}", model_name)

    with open(dst_path, "w") as f:
        f.write(content)

    # 设置权限：.sh 文件为 755，其他为 644
    if dst_path.endswith(".sh"):
        os.chmod(dst_path, 0o755)
    else:
        os.chmod(dst_path, 0o644)

    print(f"Generated file: {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a model sample from template.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing directory")
    parser.add_argument("-m", "--model", required=True, help="Path to model file (e.g., yolov5.onnx)")
    parser.add_argument("-i", "--input_cfg", help="Path to input_cfg.yml file. If not provided, auto-generate one.")

    args = parser.parse_args()

    MODEL_PATH = args.model
    INPUT_CFG_PATH = args.input_cfg
    FORCE_OVERWRITE = args.force

    if not os.path.exists(CUSTOM_DIR):
        os.makedirs(CUSTOM_DIR)
        print(f"Created directory: {CUSTOM_DIR}")

    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' does not exist.")
        sys.exit(1)

    MODEL_NAME = get_model_name(MODEL_PATH)
    TARGET_DIR = os.path.join(CUSTOM_DIR, MODEL_NAME)

    ensure_directory(FORCE_OVERWRITE, TARGET_DIR)

    print(f"Generating sample for model: {MODEL_NAME}")

    if not os.path.isdir(TEMPLATE_DIR):
        print(f"Error: Template directory '{TEMPLATE_DIR}' does not exist.")
        sys.exit(1)

    template_files = os.listdir(TEMPLATE_DIR)
    if not template_files:
        print(f"Error: No files found in template directory '{TEMPLATE_DIR}'.")
        sys.exit(1)

    for filename in template_files:
        src_path = os.path.join(TEMPLATE_DIR, filename)
        dst_path = os.path.join(TARGET_DIR, filename)

        if os.path.isfile(src_path):
            replace_model_name_and_copy(src_path, dst_path, MODEL_NAME)

    if INPUT_CFG_PATH:
        print(f"Using provided input config: {INPUT_CFG_PATH}")
        dst_cfg = os.path.join(TARGET_DIR, "input_cfg.yml")
        shutil.copy(INPUT_CFG_PATH, dst_cfg)
    else:
        print("generating default input_cfg.yml...")
        inputs_info = get_model_input_info(MODEL_PATH)
        gen_model_input_cfg(inputs_info, TARGET_DIR)

    update_cmake_file(MODEL_NAME, TOP_CMAKE_FILE)

if __name__ == "__main__":
    main()