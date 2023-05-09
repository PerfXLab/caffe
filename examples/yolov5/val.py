# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

usage example:
  cd caffe
  python examples/yolov5/val.py --rect --multi_label
"""

import os
import sys
import argparse
import ast
import tqdm
import pdb

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

sys.path.append('./examples')
from pre_process import *
from post_process import *

def readlabels(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split()
            row = [float(x) for x in row]
            data.append(row)
    return np.array(data)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
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
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros(
        (nc, 1000)), np.zeros((nc, 1000))
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
            # negative x, xp because xp decreases
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                              left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(
                    recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # IoU above threshold and classes match
    x = np.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        # [label, detection, iou]
        matches = np.concatenate(
            (np.stack(x, axis=1), iou[x[0], x[1]][:, None]), axis=1)
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = np.array(matches)
        correct[matches[:, 1].astype(int)] = matches[:, 2:3] >= iouv
    return correct


class CaffeDetection:
    def __init__(self, model_def, model_weights, image_resize,
                 output_list, norm_mean, norm_std, isgray, resize_mode,
                 rect, scaleup, auto, conf_thres, iou_thres, max_det, na, no, 
                 agnostic_nms, multi_label, strides, anchors, names):
        caffe.set_mode_cpu()

        self.image_resize = image_resize
        self.output_list = output_list
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.isgray = isgray
        self.resize_mode = resize_mode
        self.rect = rect
        self.scaleup = scaleup
        self.auto = auto
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.na = na
        self.no = no
        self.agnostic_nms = agnostic_nms
        self.multi_label = multi_label
        self.strides = strides
        self.anchors = anchors
        self.names = names

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def detect(self, image_file):
        '''
        YOLOv5 detection
        '''
        img, img0, ratio, pad = pre_process(image_file, self.image_resize,
                                self.norm_mean, self.norm_std,self.isgray, 
                                self.resize_mode, self.rect, self.scaleup, 
                                self.auto)
        print('img.shape = ',img.shape)
        ratio_pad = [ratio, pad]
        shape = img.shape
        self.net.blobs['data'].reshape(
            shape[0], shape[1], shape[2], shape[3])
        self.net.blobs['data'].data[...] = img

        # Forward pass.
        self.net.forward()
        out1 = self.output_list[0]
        out2 = self.output_list[1]
        out3 = self.output_list[2]
        result = {}
        result[out1] = self.net.blobs[out1].data
        result[out2] = self.net.blobs[out2].data
        result[out3] = self.net.blobs[out3].data

        strides = {}
        anchors = {}
        strides[out1] = self.strides[0]
        strides[out2] = self.strides[1]
        strides[out3] = self.strides[2]
        anchors[out1] = self.anchors[0]
        anchors[out2] = self.anchors[1]
        anchors[out3] = self.anchors[2]

        det = post_process(result, img, img0, self.conf_thres, self.iou_thres,
                           self.max_det, self.na, self.no, self.agnostic_nms,
                           self.multi_label, strides, ratio_pad, anchors, self.names)
        return det, img, img0, ratio, pad


def main(args):
    detection = CaffeDetection(args.model_def, args.model_weights,
                               args.image_resize, args.output_list,
                               args.norm_mean, args.norm_std, args.isgray,
                               args.resize_mode, args.rect, args.scaleup,
                               args.auto, args.conf_thres, args.iou_thres,
                               args.max_det, args.na, args.no, args.agnostic_nms,
                               args.multi_label, args.strides, args.anchors, args.names)
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = np.size(iouv)
    img_size = 640
    stride = 32
    pad = 0.5
    seen = 0
    stats = []

    image_folder_path = os.path.join(args.image_path, 'images/train2017')
    label_folder_path = os.path.join(args.image_path, 'labels/train2017')
    for filename in tqdm.tqdm(os.listdir(image_folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            print('image is = ',filename)
            img_file_path = os.path.join(image_folder_path, filename)
            label_file_path = os.path.join(label_folder_path, os.path.splitext(filename)[0]+'.txt')
            det, img, img0, ratio, pads = detection.detect(img_file_path)
     
            _, _, height, width = img.shape 
            h0, w0 = img0.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(img0, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                h, w = im.shape[:2]
                img0 = im
            h, w = h0, w0  # im, hw_original, hw_resized
            shapes = (h0, w0), ((h / h0, w / w0), pads)            
            
            labels = readlabels(label_file_path)
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pads[0], padh=pads[1])
            nl = len(labels)
            if nl:
                labels[:, 1:5] = xyxy2xywhn(
                    labels[:, 1:5], w=img.shape[-1], h=img.shape[-2], clip=True, eps=1E-3)
                labels[:, 1:] *= [width, height, width, height]
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(det) == 0:
                if nl:
                    stats.append(
                        (np.zeros((0, niou), dtype=bool), [], [], tcls))
                continue

            # Predictions
            predn = np.copy(det)
            
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img.shape[1:], tbox, shapes[0], shapes[1])  # native-space labels
                # native-space labels
                labelsn = np.concatenate((labels[:, 0:1], tbox), axis=1)
                correct = process_batch(predn, labelsn, iouv)

            else:
                correct = np.zeros((det.shape[0], niou), dtype=bool)
            stats.append(
                (correct, det[:, 4], det[:, 5], tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(args.names))
    else:
        nt = np.zeros(1)

    # Print results
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images',
                                 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')

    print(s)
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def',
                        default='examples/yolov5/yolov5s_v6.0.prototxt', type=str,
                        help='prototxt path')
    parser.add_argument(
        '--image_resize', default=[640, 640], type=int, nargs=2,
        help='image input shape[width,height]')
    parser.add_argument('--model_weights',
                        default='examples/yolov5/yolov5s_v6.0.caffemodel', type=str,
                        help='caffemodel path')
    parser.add_argument('--image_file', default='examples/yolov5/bus.jpg', type=str,
                        help='image path')
    parser.add_argument('--image_path', default='examples/yolov5/coco128', type=str,
                        help='image path')
    parser.add_argument('--isgray', action='store_true',
                        help='whether to read single-channel images')
    parser.add_argument(
        '--norm_mean', default=[0, 0, 0], type=float, nargs=3, help='normalization mean')
    parser.add_argument(
        '--norm_std', default=[255, 255, 255], type=float, nargs=3,
        help='normalization standard deviation')
    parser.add_argument('--resize_mode', default='smart', type=str,
                        help='resize modes, including none (no resize), force (resize \
                            according to image_resize), smart (resize according to the \
                            longest side of image_resize while preserving the aspect ratio)')
    parser.add_argument('--rect', action='store_true',
                        help='Whether to keep the original aspect ratio of the image')
    parser.add_argument('--scaleup', action='store_true',
                        help='Whether to scaleup the image')
    parser.add_argument('--auto', action='store_true',
                        help='The auto parameter determines the scaling method of the image')
    parser.add_argument(
        '--output_list', default=["Conv_196", "Conv_308", "Conv_420"], type=str, nargs=3,
        help='output layer name')
    parser.add_argument('--na', default=3, type=int, help='the number \
                        of anchors corresponding to each pixel point')
    parser.add_argument('--no', default=85, type=int, help='the number of categories \
        + 5 (positive sample probability and bounding box coordinates)')
    parser.add_argument('--anchors', default=[[[1.25000,  1.62500],
                                               [2.00000,  3.75000],
                                               [4.12500,  2.87500]],
                                              [[1.87500,  3.81250],
                                               [3.87500,  2.81250],
                                               [3.68750,  7.43750]],
                                              [[3.62500,  2.81250],
                                               [4.87500,  6.18750],
                                               [11.65625, 10.18750]]
                                              ], type=ast.literal_eval,
                        help='anchor values')
    parser.add_argument(
        '--strides', default=[8, 16, 32], type=int, nargs=3,
        help='feature map downsampling factor')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max_det', default=1000, type=int,
                        help='maximum detections per image')
    parser.add_argument('--agnostic_nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--multi_label', action='store_true',
                        help='multiple labels per box (adds 0.5ms/img)')
    parser.add_argument('--names', default=['person', 'bicycle', 'car', 'motorcycle',
                                            'airplane', 'bus', 'train', 'truck', 'boat',
                                            'traffic light', 'fire hydrant', 'stop sign',
                                            'parking meter', 'bench', 'bird', 'cat',
                                            'dog', 'horse', 'sheep', 'cow', 'elephant',
                                            'bear', 'zebra', 'giraffe', 'backpack',
                                            'umbrella', 'handbag', 'tie', 'suitcase',
                                            'frisbee', 'skis', 'snowboard', 'sports ball',
                                            'kite', 'baseball bat', 'baseball glove',
                                            'skateboard', 'surfboard', 'tennis racket',
                                            'bottle', 'wine glass', 'cup', 'fork',
                                            'knife', 'spoon', 'bowl', 'banana', 'apple',
                                            'sandwich', 'orange', 'broccoli', 'carrot',
                                            'hot dog', 'pizza', 'donut', 'cake', 'chair',
                                            'couch', 'potted plant', 'bed', 'dining table',
                                            'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                            'keyboard', 'cell phone', 'microwave', 'oven',
                                            'toaster', 'sink', 'refrigerator', 'book', 'clock',
                                            'vase', 'scissors', 'teddy bear', 'hair drier',
                                            'toothbrush'], nargs='+', type=str,
                        help='category names')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
