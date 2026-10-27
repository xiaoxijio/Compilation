import torch
# import mmcv
import numpy as np
import  onnxruntime
import json
from tqdm import tqdm
from torchvision.transforms.functional import rotate
import os.path as osp

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

def bbox3d2result(bboxes, scores, labels, attrs=None):
       
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

def get_bboxes(preds_dicts, rescale=False):
    """Generate bboxes from bbox head predictions.
    Args:
        preds_dicts (tuple[list[dict]]): Prediction results.
        img_metas (list[dict]): Point cloud and image's meta info.
    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """
    decoder = NMSFreeCoder(post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10)

    preds_dicts = decoder.decode(preds_dicts)
    # print('get_bboxes  preds_dicts af : ' ,len(preds_dicts) ,  preds_dicts[0])

    num_samples = len(preds_dicts)
    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds['bboxes']
        # print('bboxes : ' , bboxes.shape)

        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

        code_size = bboxes.shape[-1]

        bboxes = LiDARInstance3DBoxes(bboxes, code_size)
        # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
        scores = preds['scores']
        labels = preds['labels']

        ret_list.append([bboxes, scores, labels])

    return ret_list

def runONNX(onnx_model, can_bus_onnx, prev_bev_rotate , img , shift):
    
    outName = onnx_model.get_outputs()[0].name
    outName1 = onnx_model.get_outputs()[1].name
    outName2 = onnx_model.get_outputs()[2].name
    ins =  {
            onnx_model.get_inputs()[0].name: can_bus_onnx, \
            onnx_model.get_inputs()[1].name: prev_bev_rotate, \
            onnx_model.get_inputs()[2].name: img , \
            onnx_model.get_inputs()[3].name: shift, \
                  }
    res = onnx_model.run([outName,outName1,outName2],  ins)
    return res[0:2] , res[2]

def getShift(can_bus_af, grid_length=[2*51.2/50, 2*51.2/50] , bev_h=50 , bev_w=50):

    can_bus_af = torch.from_numpy(can_bus_af)
    delta_x = can_bus_af[0]#np.array([each[0] for each in can_bus_af])
    delta_y = can_bus_af[1]
    ego_angle = can_bus_af[-2]/np.pi * 180
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = torch.atan2(delta_y, delta_x) / np.pi * 180

    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * \
        torch.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * \
        torch.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

    shift = torch.stack((torch.unsqueeze(shift_x , 0), torch.unsqueeze(shift_y,0)) , 0).permute(1, 0)  

    return shift

def preProcess(prev_bev,  in_angle): # [2500, 1, 256]
    prev_bev = torch.from_numpy(prev_bev) # [2500, 1, 256]
    rotation_angle = float(in_angle)
    tmp_prev_bev = prev_bev[:, 0].reshape(
        50, 50, -1).permute(2, 0, 1) # [256, 50, 50]
    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,center=[100,100]) # [2500, 1, 256]
    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
        50 * 50, 1, -1) #[2500, 1, 256]
    prev_bev[:, 0] = tmp_prev_bev[:, 0] # [2500, 1, 256]
    return  prev_bev.permute(1, 0, 2).numpy() # [1, 2500, 256]

def onnxModeInfer(onnxModelPath , saveRoot, nuscenesInfo):

    providers = [
    # ('CUDAExecutionProvider', {
    #     'device_id': 0,
    #     'arena_extend_strategy': 'kNextPowerOfTwo',
    #     'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
    #     'cudnn_conv_algo_search': 'EXHAUSTIVE',
    #     'do_copy_in_default_stream': True,
    # }),
    'CPUExecutionProvider',]

    onnx_model = onnxruntime.InferenceSession(onnxModelPath , providers = providers)
    with open(nuscenesInfo, 'r', encoding='utf-8') as f:
        nuscenesInfo = json.load(f)  # 解析为 Python 对象
    tokens=[]
    for skey in nuscenesInfo.keys():
        data = nuscenesInfo[skey]
        tokens += data.keys()

    cnt = 0
    for token in tqdm(tokens):
        can_bus = np.fromfile(f"{saveRoot}/{token}/{token}_0_1x18.bin", np.float32)
        img = np.fromfile(f"{saveRoot}/{token}/{token}_2_6x3x480x800.bin" , np.float32).reshape(6,3,480,800)
        
        if cnt==0:
            tmp_pos = can_bus[0:3]
            tmp_angle = can_bus[-1]
            new_prev_bev = np.load(saveRoot+'/query.npy')
        
        rot_angle = can_bus[-1]- tmp_angle
        prev_bev_rotate = preProcess( new_prev_bev , rot_angle)
        can_bus_onnx = np.copy(can_bus)
        can_bus_onnx[0] = can_bus[0] - tmp_pos[0]
        can_bus_onnx[1] = can_bus[1] - tmp_pos[1]
        can_bus_onnx[2] = can_bus[2] - tmp_pos[2]
        can_bus_onnx[-1]  = can_bus[-1] - tmp_angle
        shift = getShift(can_bus_onnx)
        prev_bev_rotate.tofile(f"{saveRoot}/{token}/{token}_1_1x2500x256.bin")
        shift.numpy().astype(np.float32).tofile(f"{saveRoot}/{token}/{token}_3_1x2.bin")
        can_bus_onnx.tofile(f"{saveRoot}/{token}/{token}_0_1x18.bin")
        _, new_prev_bev  = runONNX(onnx_model, can_bus_onnx.astype(np.float32).reshape(1,18), prev_bev_rotate.astype(np.float32), img.astype(np.float32) , shift.numpy().astype(np.float32))
        tmp_angle = can_bus[-1]
        tmp_pos = can_bus[0:3]
        cnt = cnt + 1
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='ONNX模型路径')
    parser.add_argument('--calib_path', required=True, help='校准数据文件夹路径')
    parser.add_argument('--nuscenesInfo', required=True)
    args = parser.parse_args()
    onnxModeInfer(args.model , args.calib_path, args.nuscenesInfo)

if __name__ == "__main__":
    main()