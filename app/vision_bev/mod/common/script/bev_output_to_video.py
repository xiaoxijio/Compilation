import cv2
import os

def images_to_video(folder_path, output_path, fps=5):
    cnt = 1
    image_file = folder_path + '/' + str(1) + '/' + str(1) + "_bev_final.jpg"
    first_image = cv2.imread(image_file)
    height, width, _ = first_image.shape
    new_width = 3840
    # 计算缩放比例
    ratio = new_width / width
    # 计算新的高度
    new_height = int(height * ratio)

    print(new_width, new_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while True :
        image_file = folder_path + '/' + str(cnt) + '/' + str(cnt) + "_bev_final.jpg"
        cnt = cnt + 1
        print(image_file)
        if not os.path.exists(image_file):
            print("未找到符合要求的图片文件:", image_file)
            break
        try:
            image = cv2.imread(image_file)
            if image is not None:
                print(image_file)
                image = cv2.resize(image, (new_width, new_height))
                video_writer.write(image)
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")

    video_writer.release()

    print(f"视频已生成，保存路径为: {output_path}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', default='../../../../../data/nuscenes/5af9c7f124d84e7e9ac729fafa40ea01', help='bev result floder')
    parser.add_argument('--output_path', default='fastbev++_video.mp4', help='video save path')
    args = parser.parse_args()
    images_to_video(args.folder_path, args.output_path)
