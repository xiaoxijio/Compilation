import os, onnx
import argparse
from pathlib import Path
from onnx import helper, numpy_helper

def exetract_model(args):
    model_path = args.model          # 将得到 "a.onnx"
    input_names = args.inputs        # 将得到 ["1", "2", "3"]
    output_names = args.outputs      # 将得到 ["4", "5"]
    save_model = args.save_model     # 模型保存路径
    model = onnx.load(args.model)
    if input_names == None:
        input_names = [model.graph.input[i].name for i in range(len(model.graph.input))]
    if output_names == None:
        output_names = [model.graph.output[i].name for i in range(len(model.graph.output))]

    # 加载整个ONNX模型
    onnx.utils.extract_model(model_path, save_model, input_names, output_names)
    print("modify model has saved: ", os.path.abspath(save_model))

def insert_sigmoid_before_output(model_path):

    model = onnx.load(model_path)
    graph = model.graph

    new_nodes = []
    for output in graph.output:
        sigmoid_node = onnx.helper.make_node(
            'Sigmoid', 
            [output.name], 
            [output.name + "_sigmoid"]
        )
        new_nodes.append(sigmoid_node)
        output.name = output.name + "_sigmoid"
    graph.node.extend(new_nodes)
    onnx.save(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX模型处理工具')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='ONNX模型路径')
    
    parser.add_argument('--need_sigmoid', action="store_true", help='need insert sigmoid')
    
    parser.add_argument('-i', '--inputs', type=str, nargs='+',
                        help='输入名称列表')
    
    parser.add_argument('-o', '--outputs', type=str, nargs='+',
                        help='输出名称列表')
    
    parser.add_argument('-s', '--save_model', type=str,
                        help='输出名称列表')
    parser.add_argument('--reshape', nargs="+", type=int, default=None, help="input shape")
    args = parser.parse_args()
    exetract_model(args)
    if args.need_sigmoid:
        insert_sigmoid_before_output(args.save_model)
    