import cv2
import os
import json
import argparse

def get_all_files_abs_paths(directory):
    """
    获取目录中所有文件的绝对路径列表（包括子目录中的文件）
    :param directory: 目标目录路径
    :return: 文件绝对路径列表
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            # 构造完整路径
            filepath = os.path.join(root, filename)
            file_paths.append(os.path.abspath(filepath))
    return file_paths
def image2video(args):
    image_dir = args.image_dir
    image_files = sorted(get_all_files_abs_paths(image_dir))

    target_width = args.target_width
    target_height = args.target_height
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(args.save_path, fourcc, args.fps, (target_width, target_height), isColor=True)

    # 逐帧写入图像
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.resize(image, (target_width, target_height))
        video.write(image)

    video.release()
    print(f"Video saved to: {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default="./tmp_res/viz_results/board")
    parser.add_argument('--save-path', type=str, default="./tmp_res/viz_results/board.mp4")
    parser.add_argument('--target-width', type=int, default=1056)
    parser.add_argument('--target-height', type=int, default=784)
    parser.add_argument('--fps', type=int, default=4)
    args = parser.parse_args()
    image2video(args)