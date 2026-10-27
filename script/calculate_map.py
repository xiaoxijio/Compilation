import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

COCO_CLASS_NAME = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_box_via_dist(predi, gti):
    box_pr = predi[:, 1:].clone()
    box_gt = gti[:, 1:].clone()


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy=center, wh=width-height
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]       # width
    y[..., 3] = x[..., 3] - x[..., 1]       # height
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def get_pr_gt(pred_dir, gt_dir, nnie=False):
    '''
    pred txt format: cls x y w h conf
    gt txt format: cls x y w h
    '''
    
    pred_gt = dict()
    for txt_name in os.listdir(pred_dir):
        predi = []
        gti = []
        if os.path.exists(os.path.join(pred_dir, txt_name)):
            # try:
            pred_txt = open(os.path.join(pred_dir, txt_name))
            # except PermissionError:
            #     continue
            for line in pred_txt.readlines():
                read = np.array(list(map(float, line.rstrip().split(' '))))
                # print(f"{txt_name} det: {read}")

                # nnie
                if nnie:
                    h, w = 1024,1920
                    read_cp = read.copy()   # cls conf xmin ymin xmax ymax
                    read_cp[1:5] = read[2:6]
                    read_cp[5] = read[1]
                    read = read_cp          # cls xmin ymin xmax ymax conf
                    read[1:-1] *= 1920/960
                    read[3] -= read[1]
                    read[4] -= read[2]
                    read[1] += read[3]/2
                    read[2] += read[4]/2
                    read[[1, 3]] /= w
                    read[[2, 4]] /= h

                predi.append(read)
            pred_txt.close()
        else:predi = np.array([])
        # try:
        if os.path.exists(gt_dir+txt_name): 
            gt_txt = open(gt_dir+txt_name) if not nnie else open(gt_dir+txt_name[:-7]+"txt")
        # gt_txt = open(gt_dir+txt_name) if not nnie else open(gt_dir+txt_name[:-8]+"txt") 
        # except FileNotFoundError:
        #     continue
            for line in gt_txt.readlines():
                read = np.array(list(map(float, line.rstrip().split(' '))))
                # print(f"{txt_name} gt: {read}")
                gti.append(read)

            gt_txt.close()
        predi = torch.tensor(np.array(predi))
        gti = torch.tensor(np.array(gti))
        pred_gt[txt_name] = [predi, gti]       
    return pred_gt


def calculate_map(pred_dir, gt_dir, img_shape=[640, 640], nc=80, names=[ "car",  "bus",  "truck"], nnie=False):
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    p, r, f1, mp, mr, map50, map75, map, t0, t1, seen = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    pred_gt = get_pr_gt(pred_dir, gt_dir, nnie=nnie)
    for name in tqdm(pred_gt.keys()):
        seen += 1
        pred, gt = pred_gt[name]
        nl = len(gt)
        tcls = gt[:, 0].tolist() if nl else []  # target class
        # tcls = [t_cls for t_cls in tcls if t_cls not in [1, 5]]

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
            
        # predn = pred.clone()
        # pred[:, 1:5] = scale_coords(img_shape, predn[:, 1:5], img_shape)  # native-space pred
        # print(pred.shape)
        # pred = pred[pred[:, 3] < 80/1920]
        # gt = gt[gt[:, 3] < 60/1920]
        # print(pred.shape)
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

        if nl:
            detected = []  # target indices
            tcls_tensor = gt[:, 0]
            # tcls = [t_cls for t_cls in tcls if t_cls not in [1, 5]]

            # target boxes
            tbox = xywh2xyxy(gt[:, 1:5])
            # scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
            
            # pred[:, 2:6] = xywh2xyxy(pred[:, 2:6])
            pred[:, 1:5] = xywh2xyxy(pred[:, 1:5]) # cls x y w h conf

            # pred[:, [2, 4]] *= img_shape[0]
            # pred[:, [3, 5]] *= img_shape[1]
            # pred[:, [1, 3]] *= img_shape[0]
            # pred[:, [2, 4]] *= img_shape[1]
            # tbox[:, [0, 2]] *= img_shape[0]
            # tbox[:, [1, 3]] *= img_shape[1]

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 0]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    # ious, i = box_iou(pred[pi, 2:6], tbox[ti]).max(1)  # best ious, indices
                    ious, i = box_iou(pred[pi, 1:5], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        # stats.append((correct.cpu(), pred[:, 6], pred[:, 0], tcls))
        stats.append((correct.cpu(), pred[:, 5], pred[:, 0], tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 1 + '%12.3g' * 5  # print format
    print(pf % ('all', nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    # for i, c in enumerate(ap_class):
    #     print(pf % (names[c], nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                # if j == 0:
                #     print(np.mean(mrec[1:-1]))
                #     print(np.mean(recall[:, 0]))
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')
    # return p[:, 0], r[:, 0], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pred_dir', type=str, default="val2017_yolov7/", help='dir of prediction files (txt) cls cx cy w h conf')
    parser.add_argument('--gt_dir', type=str, default="val2017/", help='dir of target files (txt) cls cx cy w h')

    parser.add_argument('--names', type=list, default=None)
    parser.add_argument('--nnie', nargs='?', const=True, default=False)
    opt = parser.parse_args()
    opt.names = COCO_CLASS_NAME

    calculate_map(opt.pred_dir, opt.gt_dir, names=opt.names, nnie=False)

# python calculate_map --pred_dir [dir] --gt_dir [dir]
