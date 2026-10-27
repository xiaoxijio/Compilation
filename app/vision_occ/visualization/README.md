# flashocc 可视化

## 环境准备
Python环境: `evaluate`中的Python环境已经满足可视化需要的Open3D

注意：可视化需要调用OpenGL，环境需支持OpenGL，并确保相关驱动正常


## 执行可视化
```bash
    # 公版模型
    python  vis_occ_public_board.py --data_path ../../../data/flashocc  --save_path output
    # 魔改模型
    python  vis_occ_opt_board.py --data_path ../../../data/flashocc --save_path output

```