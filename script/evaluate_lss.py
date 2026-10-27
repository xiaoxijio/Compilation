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
    parser.add_argument('--data_input', type=str, required=True, help='Path to dataset')
    parser.add_argument('--data_output', type=str, required=True, help='Path to dataset')
    # config model
    parser.add_argument('--model_path', type=str, required=True, help='board model path')
    parser.add_argument('--pub', type=int, required=True, default=0, help='1 is pub model for evaluation, 0 is opt model for evaluation')
    args = parser.parse_args()

    board_runner = Board_server(args.board_ip, args.port, args.model_path)

    in_path = Path(args.data_input)
    out_path = Path(args.data_output)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    i = 0
    total = 6019
    bar_length = 50  # 进度条长度
    start_time = time.time()
    for item in in_path.iterdir():
        token_id = str(item).split('/')[-1].split('.')[0].split('_')[0]
        if args.pub == 1:
            token_id = token_id + "_6x3x128x352"
        else:
            token_id = token_id + "_6x3x256x704"
        result = board_runner.infer_once_with_file(str(item))
        result.tofile(str(out_path) + "/" + token_id + "_0.float.bin")
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
            print("\nBoard run complete.")
            return

if __name__ == '__main__':
    main()