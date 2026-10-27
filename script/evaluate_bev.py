import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from utils import Board_server

def main():
    import argparse
    parser = argparse.ArgumentParser(description='model board runner')
    # config board 
    parser.add_argument('--board_ip', type=str, required=True, help='board ip')
    parser.add_argument('--port', type=str, required=True, help='board server port')
    # config datasate
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    # config model
    parser.add_argument('--model_path', type=str, required=True, help='board model path')
    args = parser.parse_args()

    board_runner = Board_server(args.board_ip, args.port, args.model_path)

    data_path = Path(args.data_path)
    i = 0
    total = 6019
    bar_length = 50  # 进度条长度
    start_time = time.time()
    for item in data_path.iterdir():
        path_name = str(item).split('/')[-1]
        os.chmod(item, 0o777)
        os.makedirs(str(item)+"/npu_out", exist_ok=True)
        
        files = [f for f in item.iterdir() if f.is_file()]
        if len(files) == 1:
            result = board_runner.infer_once_with_file(str(files[0]))
            outfile0 = result[0:2*128*128]
            outfile0.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_0.float.bin")
            outfile1 = result[2*128*128:3*128*128]
            outfile1.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_1.float.bin")
            outfile2 = result[3*128*128:6*128*128]
            outfile2.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_2.float.bin")
            outfile3 = result[6*128*128:8*128*128]
            outfile3.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_3.float.bin")
            outfile4 = result[8*128*128:10*128*128]
            outfile4.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_4.float.bin")
            outfile5 = result[10*128*128:20*128*128]
            outfile5.tofile(str(item) + "/npu_out/" +path_name+ "_6x3x256x704_5.float.bin")
        i += 1
        # 打印进度条
        progress = (i + 1) / total
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        if i > 0:
            remaining = (elapsed / (i + 1)) * (total - i - 1)
            remaining_str = str(timedelta(seconds=int(remaining)))
        else:
            remaining_str = "?"
        
        sys.stdout.write(
            f"\r[{bar}] {int(progress * 100)}% | "
            f"已用: {elapsed_str} | 剩余: {remaining_str}"
        )
        sys.stdout.flush()
        if i == total: 
            print("\nEvaluate complete.")
            return

if __name__ == '__main__':
    main()