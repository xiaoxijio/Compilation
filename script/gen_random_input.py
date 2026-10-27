import numpy as np
import onnxruntime as ort
import argparse
import os

def generate_random_inputs(onnx_model_path, output_dir):
    sess = ort.InferenceSession(onnx_model_path)
    inputs = sess.get_inputs()

    os.makedirs(output_dir, exist_ok=True)

    for idx, input_info in enumerate(inputs):
        name = input_info.name
        shape = input_info.shape
        dtype = input_info.type

        if dtype == 'tensor(float)':
            np_dtype = np.float32
        elif dtype == 'tensor(int64)':
            np_dtype = np.int64
        elif dtype == 'tensor(int32)':
            np_dtype = np.int32
        else:
            raise ValueError(f"Unsupported input type: {dtype}")

        fixed_shape = [d if isinstance(d, int) else 1 for d in shape]
        fixed_shape = [1 if d == -1 else d for d in fixed_shape]

        data = np.random.rand(*fixed_shape).astype(np_dtype)

        file_name = f"{idx}.bin"
        file_path = os.path.join(output_dir, file_name)
        data.tofile(file_path)
        print(f"Saved input '{name}' to {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate random input files for onnx model.")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--output", required=True, help="Directory to save the generated .bin files")

    args = parser.parse_args()
    generate_random_inputs(args.model, args.output)
