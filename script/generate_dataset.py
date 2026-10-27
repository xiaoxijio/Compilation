# python generate_dataset.py --calib_path tmp_res/after_preprocess/ --save_path dataset.txt
# 按照给定的路径，生成校验集的文件列表
# 如果给定文件则为单输入校验集
# 如果给定文件夹则为多输入校验集

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_path', required=True, help='校准数据文件夹路径')
    parser.add_argument('--save_path', required=True, help='保存数据集列表的文件路径')
    args = parser.parse_args()

    calib_path = os.path.abspath(args.calib_path)
    save_path = args.save_path

    # 判断是文件夹下全是文件，还是全是文件夹
    items = [os.path.join(calib_path, f) for f in os.listdir(calib_path)]
    files = [f for f in items if os.path.isfile(f)]
    dirs = [f for f in items if os.path.isdir(f)]

    with open(save_path, 'w') as f:
        if len(files) > 0 and len(dirs) == 0:
            # 只有文件，全部写入，每行一个绝对路径
            for file_path in sorted(files):
                f.write(file_path + '\n')
            print("generate calib dataset:", save_path, "success")
        elif len(dirs) > 0:
            # 只有文件夹，每个文件夹一行，行内是该文件夹下所有文件的绝对路径，空格分隔
            for dir_path in sorted(dirs, key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else x):
                sub_files = [os.path.join(dir_path, sf) for sf in sorted(os.listdir(dir_path))]
                sub_files = [sf for sf in sub_files if os.path.isfile(sf)]
                if sub_files:
                    f.write(' '.join(sub_files) + '\n')
            print("generate calib dataset:", save_path, "success")
        else:
            print("calib_path:", calib_path, "err")

if __name__ == "__main__":
    main()