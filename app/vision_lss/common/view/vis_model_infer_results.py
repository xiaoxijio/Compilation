from pathlib import Path
import numpy as np
import os
import cv2
import pickle
from fire import Fire


def gen_dx_bx(xbound, ybound, zbound):
    # 将输入打包为数组，便于向量化操作
    bounds = np.array([xbound, ybound, zbound])
    
    # dx: 每个维度的步长（分辨率）
    dx = bounds[:, 2]  # 第三列是步长
    
    # bx: 每个维度的起始位置 + 半个步长（即第一个格子中心）
    bx = bounds[:, 0] + dx / 2.0
    
    # nx: 每个维度的格子数量（向上取整或整数除法）
    nx = ((bounds[:, 1] - bounds[:, 0]) / dx).astype(np.int64)  # 转为整数

    return dx, bx, nx


def draw_legend_on_image(
    image,
    legend_info,
    position,
    img_h,
    img_w,
    box_padding=10,       # 图例框内边距 (像素)
    item_spacing=15,      # 图例项之间的垂直间距 (像素)
    color_patch_size=(30, 10), # 颜色块大小 (宽, 高)
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    font_color=(0, 0, 0), # 白色文字
    font_thickness=1,
    box_color=(255, 255, 255),     # 图例框背景色 (深灰色)
    box_alpha=0.0               # 图例框透明度 (0.0 - 1.0)
):
    """
    在 OpenCV 图像上手动绘制图例。

    Args:
        image (numpy.ndarray): 要绘制图例的图像 (BGR 格式)。
        legend_info (list of dict): 图例项信息列表。
            每个字典应包含 'color' 和 'label' 键。
            'color' 可以是:
                - (B, G, R) 元组 (0-255 整数)
                - (R, G, B, A) 元组 (0.0-1.0 浮点数，A会被忽略)
                - '#RRGGBB' 字符串 (会被转换为 BGR)
            'label' 是字符串。
        position (tuple): 图例框左上角的归一化位置 (x_ratio, y_ratio)。
        box_padding (int): 图例框内边距。
        item_spacing (int): 图例项之间的垂直间距。
        color_patch_size (tuple): 颜色块的尺寸 (宽, 高)。
        font: OpenCV 字体类型。
        font_scale (float): 字体大小。
        font_color (tuple): 字体颜色 (B, G, R)。
        font_thickness (int): 字体粗细。
        box_color (tuple): 图例框背景色 (B, G, R)。
        box_alpha (float): 图例框背景透明度 (0.0 = 完全透明, 1.0 = 不透明)。

    Returns:
        numpy.ndarray: 绘制了图例的图像。
    """
    img_h, img_w = image.shape[:2]

    # 1. 计算图例框在像素坐标下的起始位置
    start_x = position[0]
    start_y = position[1]

    # 2. 计算图例项所需的最大文本宽度和总高度，以确定图例框大小
    max_text_width = 0
    text_height = 0
    for item in legend_info:
        label = item['label']
        # getTextSize 返回 (width, height) 和 baseline
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_w)
        text_height = max(text_height, text_h)

    # 计算图例框的尺寸
    patch_w, patch_h = color_patch_size
    # 图例框宽度 = 左内边距 + 色块宽 + 色块与文本间距 + 最大文本宽 + 右内边距
    legend_width = box_padding + patch_w + box_padding // 2 + max_text_width + box_padding
    # 图例框高度 = 上内边距 + N * (max(色块高, 文本高) + 间距) - 最后一项的间距 + 下内边距
    num_items = len(legend_info)
    legend_height = box_padding + num_items * (max(patch_h, text_height) + item_spacing) - item_spacing + box_padding

    # 3. 绘制半透明背景框 (如果 alpha < 1.0)
    if 0.0 <= box_alpha < 1.0:
        # 创建一个与原图相同尺寸的覆盖层
        overlay = image.copy()
        end_x_box = min(start_x + legend_width, img_w)
        end_y_box = min(start_y + legend_height, img_h)
        # 在覆盖层上绘制矩形
        cv2.rectangle(overlay, (start_x, start_y), (end_x_box, end_y_box), box_color, -1) # -1 表示填充
        # 将覆盖层与原图按 alpha 混合
        cv2.addWeighted(overlay, box_alpha, image, 1 - box_alpha, 0, image)
    elif box_alpha == 1.0:
         end_x_box = min(start_x + legend_width, img_w)
         end_y_box = min(start_y + legend_height, img_h)
         cv2.rectangle(image, (start_x, start_y), (end_x_box, end_y_box), box_color, -1)

    # 4. 绘制每个图例项
    current_y = start_y + box_padding # 当前项绘制的 Y 起始位置
    for item in legend_info:
        label = item['label']
        color_spec = item['color']

        # 解析颜色
        if isinstance(color_spec, str) and color_spec.startswith('#'):
            # 处理 '#RRGGBB' 格式 (转换为 BGR)
            hex_color = color_spec.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            bgr_color = (rgb[2], rgb[1], rgb[0]) # 转换为 BGR
        elif isinstance(color_spec, (tuple, list)) and len(color_spec) in [3, 4]:
            if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in color_spec[:3]):
                # 处理 (R, G, B, A) 或 (R, G, B) 0.0-1.0 格式
                bgr_color = tuple(int(c * 255) for c in reversed(color_spec[:3])) # 转换为 0-255 并变为 BGR
            elif all(isinstance(c, int) and 0 <= c <= 255 for c in color_spec[:3]):
                # 处理 (B, G, R) 或 (B, G, R, A) 0-255 格式
                bgr_color = tuple(color_spec[:3]) # 假设已经是 BGR
            else:
                # 默认或错误处理
                print(f"Warning: Color format not fully recognized for {color_spec}. Using default white.")
                bgr_color = (255, 255, 255)
        else:
             print(f"Warning: Unknown color format {color_spec}. Using default white.")
             bgr_color = (255, 255, 255)


        # 绘制颜色块
        patch_x1 = start_x + box_padding
        patch_y1 = current_y
        patch_x2 = min(patch_x1 + patch_w, img_w)
        patch_y2 = min(patch_y1 + patch_h, img_h)
        cv2.rectangle(image, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color, -1) # -1 填充

        # 绘制文本
        text_x = patch_x2 + box_padding // 2
        # 文本基线对齐到色块中心附近
        text_y = patch_y1 + (patch_h + text_height) // 2 + baseline // 2
        cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # 更新下一个项的 Y 位置
        current_y += max(patch_h, text_height) + item_spacing

    return image

def draw_poly(canvas, bx, dx, scale, start_x, start_y, counter, lmap, out_size):
    try:
        poly_names = ['road_segment', 'lane']
        # 定义颜色 (BGR)
        map_fill_color = (int(195), int(205), int(248)) # (B, G, R) ~ (1.00, 0.50, 0.31) alpha=0.2

        # 绘制多边形 (填充)
        for name in poly_names:
            for la in lmap.get(name, []):
                # 转换坐标 (假设 la 是 Nx2 的点数组)
                pts = (la - bx[:2]) / dx[:2] # 只使用 x, y
                pts[:, 1] = out_size - pts[:, 1] # 如果需要精确翻转
                pts[:, 0] = out_size - pts[:, 0] # 如果需要精确翻转
                
                pts_canvas = pts.copy()
                pts_canvas1 = pts.copy()
                pts_canvas1[:, 1] = pts_canvas[:, 0] * scale + start_y # x
                pts_canvas1[:, 0] = pts_canvas[:, 1] * scale + start_x # y
                pts_canvas = pts_canvas1.round().astype(np.int32)
                if len(pts_canvas) > 2:
                    cv2.fillPoly(canvas, [pts_canvas], color=map_fill_color)
    except Exception as e:
        print(f"Warning: Could not draw map for sample {counter}: {e}")

def draw_road_and_lane(canvas, bx, dx, scale, start_x, start_y, counter, lmap, out_size):
    try:

        # 定义颜色 (BGR)
        road_divider_color = (int(1.0 * 255), int(0.0 * 255), int(0.0 * 255)) # (B, G, R) ~ (0.0, 0.0, 1.0) alpha=0.5
        lane_divider_color = (159, 0, 255) # (B, G, R)

        # 绘制线条 (道路分隔线)
        for la in lmap.get('road_divider', []):
            pts = (la - bx[:2]) / dx[:2]
            pts[:, 1] = out_size - pts[:, 1] # 如果需要精确翻转
            pts[:, 0] = out_size - pts[:, 0] # 如果需要精确翻转
            pts_canvas = pts.copy()
            pts_canvas1 = pts.copy()
            pts_canvas1[:, 1] = pts_canvas[:, 0] * scale + start_y
            pts_canvas1[:, 0] = pts_canvas[:, 1] * scale + start_x
            pts_canvas = pts_canvas1.round().astype(np.int32)
            if len(pts_canvas) > 1:
                # 绘制折线
                for i in range(len(pts_canvas) - 1):
                    pt1 = tuple(pts_canvas[i])
                    pt2 = tuple(pts_canvas[i+1])
                    cv2.line(canvas, pt1, pt2, color=road_divider_color, thickness=1, lineType=cv2.LINE_AA)

        # 绘制线条 (车道分隔线)
        for la in lmap.get('lane_divider', []):
            pts = (la - bx[:2]) / dx[:2]
            pts[:, 1] = out_size - pts[:, 1] # 如果需要精确翻转
            pts[:, 0] = out_size - pts[:, 0] # 如果需要精确翻转
            pts_canvas = pts.copy()
            pts_canvas1 = pts.copy()
            pts_canvas1[:, 1] = pts_canvas[:, 0] * scale + start_y
            pts_canvas1[:, 0] = pts_canvas[:, 1] * scale + start_x
            pts_canvas = pts_canvas1.round().astype(np.int32)
            if len(pts_canvas) > 1:
                for i in range(len(pts_canvas) - 1):
                    pt1 = tuple(pts_canvas[i])
                    pt2 = tuple(pts_canvas[i+1])
                    cv2.line(canvas, pt1, pt2, color=lane_divider_color, thickness=1, lineType=cv2.LINE_AA)

    except Exception as e:
        print(f"Warning: Could not draw map for sample {counter}: {e}")

def draw_ego(canvas, start_x, start_y, new_width, new_height, scale, bx, dx, out_size):
    legend_items = [
        {'color': (0.0, 0.0, 1.0, 1.0), 'label': 'Output Vehicle Segmentation'}, 
        {'color': '#76b900', 'label': 'Ego Vehicle'},                          
        {'color': (1.00, 0.50, 0.31, 0.8), 'label': 'Map (for visualization purposes only)'} 
    ]
    if out_size == 200:
        draw_legend_on_image(canvas, legend_items, (start_x, start_y), new_width, new_height)
    else:
        draw_legend_on_image(canvas, legend_items, (start_x, start_y), new_width, new_height,color_patch_size=(60, 20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7)


    # --- 绘制自车 (Ego Vehicle) ---
    # 定义自车矩形顶点 (在 BEV 空间中)
    W_ego = 1.85
    L_ego = 4.084
    pts_ego = np.array([
        [-L_ego/2. + 0.5,  W_ego/2.],  # 后左
        [ L_ego/2. + 0.5,  W_ego/2.],  # 前左
        [ L_ego/2. + 0.5, -W_ego/2.],  # 前右
        [-L_ego/2. + 0.5, -W_ego/2.],  # 后右
    ])
    
    # 转换坐标 (假设 bx, dx 是 4 维，取前 2 维用于 x,y)
    pts_ego_map = (pts_ego - bx) / dx

    pts_ego_map[:, 1] = out_size - pts_ego_map[:, 1] # 如果需要精确翻转
    pts_ego_map[:, 0] = out_size - pts_ego_map[:, 0] # 如果需要精确翻转

    # 调整到画布坐标 (仅顶部区域)
    offset_x = start_x
    offset_y = start_y
    pts_ego_canvas = pts_ego_map.copy()
    pts_ego_canvas1 = pts_ego_map.copy()
    pts_ego_canvas1[:, 1] = pts_ego_canvas[:, 0] * scale  + offset_y
    pts_ego_canvas1[:, 0] = pts_ego_canvas[:, 1] * scale  + offset_x
    pts_ego_canvas = pts_ego_canvas1.round().astype(np.int32)
    # 绘制填充多边形 (BGR color #76b900 -> approx (0, 185, 118))
    ego_color = (0, 185, 118) # (B, G, R)
    cv2.fillPoly(canvas, [pts_ego_canvas], color=ego_color)


def viz_model_preds(save_folder='./viz_results',
                    dataroot='./data/nuscenes',
                    H=900, W=1600,
                    final_dim=(256, 704),
                    board_outputs=None,
                    float_outputs=None,
                    gt_outputs=None,
                    pkl_path=None
                    ):

    if board_outputs is not None:
        outputs = board_outputs
    elif float_outputs is not None:
        outputs = float_outputs
    elif gt_outputs is not None:
        outputs = gt_outputs
    else:
        return

    # 图片裁剪参数
    fH, fW = final_dim
    resize = max(fH/H, fW/W)
    resize_dims = (int(W*resize), int(H*resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean((0.0, 0.22)))*newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    if fH == 128 :
        grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [4.0, 45.0, 1.0],
        }

        out_size = 200
    else:

        grid_conf = {
            'xbound': [-51.2,51.2,0.8],
            'ybound': [-51.2,51.2,0.8],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [4.0, 45.0, 1.0],
        }
        out_size = 125

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2], bx[:2]

    dpi = 100
    val = 0.01
    fH, fW = final_dim
    # 计算画布尺寸 (像素)
    canvas_width = int(3 * fW * val * dpi)
    canvas_height = int((1.5 * fW + 2 * fH) * val * dpi)
    

    # 定义子图画布区域 (y1, y2, x1, x2) 像素坐标
    # gs[0, :] 顶部大图
    top_view_y1 = 0
    top_view_y2 = int(1.5 * fW * val * dpi) # 高度
    top_view_x1 = 0
    top_view_x2 = canvas_width # 宽度

    # gs[1 + imgi // 3, imgi % 3] 底部 3x3 网格 (只用前6个)
    bottom_grid_y1 = top_view_y2
    bottom_grid_y2 = canvas_height
    bottom_grid_x1 = 0
    bottom_grid_x2 = canvas_width

    grid_cell_h = (bottom_grid_y2 - bottom_grid_y1) // 2 # 2行
    grid_cell_w = (bottom_grid_x2 - bottom_grid_x1) // 3 # 3列

    counter = 0

    # 定义要创建的文件夹路径
    folder_path = Path(save_folder)

    # 创建文件夹，如果已存在则不报错
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, 'rb') as f:
        pkl_dict = pickle.load(f)

    lmap_dict = pkl_dict["lmap_dict"]
    imgs_path_dict = pkl_dict["imgs_path_dict"]
    filename = ""
    for batchi, imgs_path in enumerate(imgs_path_dict.items()):
        imgs_path = imgs_path[1]
        if final_dim[0] == 128:
            B, N, C, H, W = 1, 6, 3, 128, 352
        else:
            B, N, C, H, W = 1, 6, 3, 256, 704
        if board_outputs is not None:
            filename = os.path.join(outputs, f'{counter}_{N}x{C}x{H}x{W}_0.float.bin')
        else:
            filename = os.path.join(outputs, f'{counter}_{N}x{C}x{H}x{W}.bin')
        if not os.path.exists(filename):
            return
        if H == 256:
            out = np.fromfile(filename, dtype=np.float32).reshape(1, 1, 128, 128)
        else:
            out = np.fromfile(filename, dtype=np.float32).reshape(1, 1, 200, 200)
        
        out_sigmoid = 1.0 / (1.0 + np.exp(-out))
        out_map = out_sigmoid[0].squeeze(0) # Shape: (H_out, W_out)

        # --- 创建画布 ---
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        # --- 绘制顶部 BEV 输出 ---
        # 准备 BEV 图像数据 (归一化到 0-255)
        bev_img = (out_map * 255).astype(np.uint8)

        blue_palette = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            blue_palette[i] = [int(200 + (i / 255.0) * 55), 0, 0] if i>128 else [0, 0, 0] # BGR 格式，这里固定 G 和 R 为 0，只改变 B
        # blue_palette = cv2.COLORMAP_WINTER
        # 应用颜色映射
        bev_colored = cv2.applyColorMap(bev_img, blue_palette)

        # 调整 BEV 图像大小以适应顶部区域
        canvas_w = top_view_x2 - top_view_x1 
        canvas_h = top_view_y2 - top_view_y1
        h, w = bev_colored.shape[:2]
        scale = canvas_h / h
        new_width = int(w * scale)
        new_height = int(h * scale) # 这应该等于 canvas_h
        bev_resized = cv2.resize(bev_colored, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # 翻转图片
        bev_resized = cv2.flip(bev_resized, flipCode=1)
        bev_resized = cv2.flip(bev_resized, flipCode=0)
        start_x = max(0, (canvas_w - new_width) // 2)
        start_y = 0 # 高度已对齐，Y 起始点为 0

        end_x = min(start_x + new_width, canvas_w)
        end_y = min(start_y + new_height, canvas_h)

        # --- 绘制地图元素 (需要 bx, dx, etc.) ---
        scale = canvas_h / out_size
        draw_poly(canvas, bx, dx, scale, start_x, start_y, counter, lmap_dict[batchi], out_size)
        
        # --- 绘制周围车辆 ---
        mask_non_white = ~(np.all(bev_resized == [0, 0, 0], axis=-1))

        mask_3d = np.repeat(mask_non_white[:, :, np.newaxis], 3, axis=2)
        canvas[start_y:end_y, start_x:end_x][mask_3d] = bev_resized[mask_3d]

        draw_road_and_lane(canvas, bx, dx, scale, start_x, start_y, counter, lmap_dict[batchi], out_size)

        # --- 绘制底部 6 个摄像头图像 ---
        for imgi, img_path in enumerate(imgs_path): # 假设最多绘制6个摄像头

            # 读取图像
            img_path = img_path.replace("./data/nuscenes",dataroot)
            img_cv = cv2.imread(img_path)
            img_cv = cv2.resize(img_cv, resize_dims, interpolation=cv2.INTER_LINEAR)

            # 1. 裁剪 (crop) - 对应 PIL 的 crop
            left, top, right, bottom = crop
            img_cv = img_cv[top:bottom, left:right]
            # 2. 如果是下半部分图像 (imgi > 2)，进行翻转
            if imgi > 2:
                    img_cv = cv2.flip(img_cv, 1)
            # 3. 转换为 OpenCV 格式 (BGR)
            # img_cv = cv2.cvtColor(np.array(img_cv), cv2.COLOR_RGB2BGR)
            # 4. 调整图像大小以适应网格单元
            img_cv_resized = cv2.resize(np.array(img_cv), (grid_cell_w, grid_cell_h))
            # 5. 计算放置位置
            row = 1 if imgi > 2 else 0 # imgi 0,1,2 -> row 0; 3,4,5 -> row 1
            col = imgi % 3
            y1 = bottom_grid_y1 + row * grid_cell_h
            y2 = y1 + grid_cell_h
            x1 = bottom_grid_x1 + col * grid_cell_w
            x2 = x1 + grid_cell_w
            # 6. 放置图像到画布
            canvas[y1:y2, x1:x2] = img_cv_resized

            # 7. 添加摄像头名称 (可选)
            cam_name = cams[imgi].replace('_', ' ')
            text_org = (x1 + 5, y1 + 20) # 文本起始点
            if out_size == 200:
                cv2.putText(canvas, cam_name, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA) # 白色文字
            else:
                cv2.putText(canvas, cam_name, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA) # 白色文字
                

        left_white = np.zeros((end_y, start_x, 3), dtype=np.uint8)
        left_white[:] = (255, 255, 255)
        canvas[0:end_y,0:start_x] = left_white

        right_white = np.zeros((end_y, start_x, 3), dtype=np.uint8)
        right_white[:] = (255, 255, 255)
        canvas[0:end_y,end_x-1:-1] = left_white

        top_left = (start_x, start_y)
        bottom_right = (end_x, end_y)

        # 绘制矩形 (绿色，线宽 2)
        cv2.rectangle(canvas, top_left, bottom_right, color=(255, 0, 0), thickness=2)


        #绘制图标
        draw_ego(canvas, start_x, start_y, new_width, new_height, scale, bx, dx, out_size)

        # --- 保存图像 ---
        si = 0
        imname = f'eval{batchi:06}_{si:03}.jpg'
        impath = os.path.join(folder_path, imname)
        print(f'saving {impath}')
        success = cv2.imwrite(impath, canvas)
        if not success:
            print(f"Error: Failed to save image {impath}")

        counter += 1

def merge_images_cv2(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    files1 = {f for f in os.listdir(folder1) if os.path.splitext(f.lower())[1] in exts}
    files2 = {f for f in os.listdir(folder2) if os.path.splitext(f.lower())[1] in exts}
    common_files = files1 & files2

    for filename in common_files:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)

            if img1 is None or img2 is None:
                print(f"⚠️ 读取失败: {filename}")
                continue

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h = max(h1, h2)

            # 调整高度
            if h1 != max_h:
                img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
            if h2 != max_h:
                img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))

            # 横向拼接
            combined = np.hstack([img1, img2])
            cv2.imwrite(output_path, combined)
            print(f"✅ 合并完成: {filename}")

        except Exception as e:
            print(f"❌ 失败: {filename}, 错误: {e}")

def viz_model_preds_process(save_folder='./viz_results',
                    dataroot='./data/nuscenes',
                    H=900, W=1600,
                    final_dim=(128,354),
                    board_outputs=None,
                    float_outputs=None,
                    gt_outputs=None,
                    pkl_path=None
                    ):
    if board_outputs is not None:
        print("board_outputs")
        viz_model_preds(save_folder=os.path.join(save_folder,"board"),
                    dataroot=dataroot,
                    H=H, W=W,
                    final_dim=final_dim,
                    board_outputs=board_outputs,
                    float_outputs=None,
                    gt_outputs=None,
                    pkl_path=pkl_path
                    )
    if float_outputs is not None:
        print("float_outputs")
        viz_model_preds(save_folder=os.path.join(save_folder,"float"),
                    dataroot=dataroot,
                    H=H, W=W,
                    final_dim=final_dim,
                    board_outputs=None,
                    float_outputs=float_outputs,
                    gt_outputs=None,
                    pkl_path=pkl_path
                    ) 
    if gt_outputs is not None:
        print("gt_outputs")
        viz_model_preds(save_folder=os.path.join(save_folder,"gt"),
                    dataroot=dataroot,
                    H=H, W=W,
                    final_dim=final_dim,
                    board_outputs=None,
                    float_outputs=None,
                    gt_outputs=gt_outputs,
                    pkl_path=pkl_path
                    )
    if board_outputs is not None and float_outputs is not None:
        merge_images_cv2(os.path.join(save_folder,"float"), os.path.join(save_folder,"board"), os.path.join(save_folder,"float_and_board"))
    if board_outputs is not None and gt_outputs is not None:
        merge_images_cv2(os.path.join(save_folder,"gt"), os.path.join(save_folder,"board"), os.path.join(save_folder,"gt_and_board"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default='./tmp_res/viz_results', help='Folder to save visualizations')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help='Root directory of the dataset')
    parser.add_argument('--H', type=int, default=900, help='Original image height')
    parser.add_argument('--W', type=int, default=1600, help='Original image width')
    parser.add_argument('--final_dim', type=int, nargs=2, default=(128, 352), help='Final dimension (height width)')
    parser.add_argument('--board_outputs', type=str, default='./data/outputs', help='Path to board outputs directory')
    parser.add_argument('--float_outputs', type=str, default=None, help='Path to float outputs directory')
    parser.add_argument('--gt_outputs', type=str, default=None, help='Path to ground truth outputs directory')
    parser.add_argument('--pkl_path', type=str, default='../common/view/vis_img_path_and_map_30.pkl', help='Path to the pickle file containing lmap_dict and imgs_path_dict')
    parser.add_argument('--pub', type=str, default=1, help='model for pub or opt')
    args = parser.parse_args()

    if args.pub == '0':
        args.final_dim = (256, 704)
    
    viz_model_preds_process(
        save_folder=args.save_folder,
        dataroot=args.dataroot,
        H=args.H,
        W=args.W,
        final_dim=tuple(args.final_dim),
        board_outputs=args.board_outputs,
        float_outputs=args.float_outputs,
        gt_outputs=args.gt_outputs,
        pkl_path=args.pkl_path
    )

if __name__ == '__main__':
    main()


                        