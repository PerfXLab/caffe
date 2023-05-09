import numpy as np
import time
import cv2
from itertools import product
import math as m

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=1):
    # Calculate the maximum value of each row
    row_max = x.max(axis=axis)

    # Each element in the row needs to be subtracted by the corresponding
    # maximum value, otherwise calculating exp(x) will overflow, resulting
    # in an inf situation
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # Calculate the exponential power of e
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def make_grid(anchors, stride, nx=20, ny=20, na=3):
    xv, yv = np.meshgrid([i for i in range(nx)], [i for i in range(ny)])
    grid = np.tile(np.stack((xv, yv), 2), (1, na, 1, 1, 1)).astype(np.float32)
    anchor_grid = np.tile(np.reshape(np.array(
        anchors) * stride, (1, na, 1, 1, 2)), (1, 1, ny, nx, 1)).astype(np.float32)
    return grid, anchor_grid


def nms(bounding_boxes, confidence_score, threshold):
    # Returns the index of the data retained after NMS.

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    #start_x = boxes[:, 0] - boxes[:, 2]/2.0
    #start_y = boxes[:, 1] - boxes[:, 3]/2.0
    #end_x = boxes[:, 0] + boxes[:, 2]/2.0
    #iend_y = boxes[:, 1] + boxes[:, 3]/2.0
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    # picked_boxes = []
    # picked_score = []
    picked_index = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    if order.ndim == 2:
        order = np.squeeze(order, axis=0)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        # picked_boxes.append(bounding_boxes[index])
        # picked_score.append(confidence_score[index])
        picked_index.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        
        # Compute the ratio between intersection and union
        ratio = intersection / \
            (areas[index] + areas[order[:-1]] - intersection)
        
        left = np.where(ratio < threshold)
        order = order[left]
    return np.array(picked_index)  # picked_index


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
    inter = np.minimum(box1[:, None, 2:], box2[:, 2:]) - \
        np.maximum(box1[:, None, :2], box2[:, :2])
    inter = np.clip(inter, 0, np.inf)
    inter = np.prod(inter, axis=2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right

    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# yolo
def non_max_suppression_yolo(prediction, conf_thres=0.25, iou_thres=0.45,
                             agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
       Implementation of output processing and NMS

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    labels = ()

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)
            ((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            #i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            i, j = np.nonzero(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j+5, None], j[:,None].astype(np.float32)), axis=1)
        else: # best class only
            conf = np.max(x[:, 5:], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:], axis=1).reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[
                conf.reshape(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).astype(np.float32) \
                / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


# ssd
def non_max_suppression_ssd(class_pred, box_pred, anchors,
                            conf_thres=0.25, iou_thres=0.45, 
                            topk=200, max_det=100):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]

    class_p = np.transpose(class_p, (1, 0))  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max = class_p.max(axis=0)  # [19248]
    class_p_max_index = np.argmax(class_p, axis=0)

    # filter predicted boxes according the class score
    keep = (class_p_max > conf_thres)
    if len(np.where(keep)[0]) > topk:
        class_p_max_thre = class_p_max[keep, :]
        sorted_indices = np.argsort(-class_p_max_thre)
        keep = sorted_indices[:200]
    class_thre = class_p[:, keep]
    box_thre, anchor_thre = box_p[keep, :], anchors[keep, :]
    cls = class_p_max_index[keep]
    
    # decode boxes
    box_thre = np.concatenate((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                               anchor_thre[:, 2:] * np.exp(box_thre[:, 2:] * 0.2)), axis=1)
    box_coord = xywh2xyxy(box_thre[:, :4])
    
    i = nms(box_coord, class_thre, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    output = [np.zeros((0, 6))]
    output[0] = np.concatenate((np.take(box_coord, i, axis=0),
                             np.take(class_thre.reshape(-1,1),i,axis=0),
                             np.take(cls.reshape(class_thre.shape).reshape(-1,1),i,axis=0)),axis=1)
    return output


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
               '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
               '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
               'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def make_anchors(img_h, img_w, conv_h, conv_w, scale, aspect_ratios=[1, 0.5, 2]):
    """Generate anchors for the SSD series model based on the size of the output 
    conv of each layer and the aspect ratio

    """
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = m.sqrt(ar)
            w = scale * ar / img_w
            h = scale / ar / img_w

            prior_data += [x, y, w, h]

    return prior_data


def visualization(detection, img, names):
    # create a color cycle
    colors = Colors()  # create instance for 'from utils.plots import colors'

    for *xyxy, conf, cls in reversed(detection):
        print('result is ',xyxy,conf,cls)
        x1, y1, x2, y2 = [round(x) for x in xyxy]
        class_name = f"{names[int(cls)]}"
        color = colors(int(cls), True)
        thickness = max(round(sum(img.shape) / 2 * 0.003), 2)
        txt_color = (255, 255, 255)
        text = f"{class_name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color,
                      thickness, lineType=cv2.LINE_AA)
        if text:
            tf = max(thickness - 1, 1)  # font thickness
            # text width, height
            w, h = cv2.getTextSize(
                text, 0, fontScale=thickness / 3, thickness=tf)[0]
            p1 = (x1, y1)
            ps = (x2, y2)
            outside = p1[1] - h - 3 >= 0  # text fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0, thickness / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    cv2.imwrite("./result.jpg", img)


# YOLO series
def post_process_yolo(result, img, img0, conf_thres=0.25, iou_thres=0.45,
                      max_det=1000, na=3, no=85, agnostic_nms=False,
                      multi_label=False, visual=False, strides=[8, 16, 32], 
                      ratio_pad=None,
                      anchors=[[[1.25000,  1.62500],
                                [2.00000,  3.75000],
                                [4.12500,  2.87500]],
                               [[1.87500,  3.81250],
                                [3.87500,  2.81250],
                                [3.68750,  7.43750]],
                               [[3.62500,  2.81250],
                                [4.87500,  6.18750],
                                [11.65625, 10.18750]]],
                      names=['person', 'bicycle', 'car', 'motorcycle',
                             'airplane', 'bus', 'train', 'truck', 'boat',
                             'traffic light', 'fire hydrant', 'stop sign',
                             'parking meter', 'bench', 'bird', 'cat', 'dog',
                             'horse', 'sheep', 'cow', 'elephant', 'bear',
                             'zebra', 'giraffe', 'backpack', 'umbrella',
                             'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                             'snowboard', 'sports ball', 'kite', 'baseball bat',
                             'baseball glove', 'skateboard', 'surfboard',
                             'tennis racket', 'bottle', 'wine glass', 'cup',
                             'fork', 'knife', 'spoon', 'bowl', 'banana',
                             'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                             'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                             'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                             'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                             'book', 'clock', 'vase', 'scissors', 'teddy bear',
                             'hair drier', 'toothbrush']):
    '''
    Runs post_process on inference results

    Inputs:
        "result":   The output of net.forward(), if len(result) > 1, you need to organize
        the result into a dictionary according to the output name of the layer.
        "img": Input original image.
        "img0": Input original image.
        "conf_thres": Confidence threshold, which is 0.25 by default.
        "iou_thres": NMS IoU threshold, which is 0.45 by default.
        "max_det": Maximum detections per image, which is 1000 by default.
        "na":  This is a parameter of the object detection model that represents the number
        of anchors corresponding to each pixel point, which is 3 by default.
        "no":  The parameters of the object detection model represent the number of
        categories + 5 (positive sample probability and bounding box coordinates), which
        is 85 by default for the YOLO series.
        "agnostic_nms": class-agnostic NMS. Class-agnostic NMS (Non-Maximum Suppression)
        is a technique used in object detection algorithms. Its purpose is to eliminate
        redundant detection results by only retaining the one with the highest confidence
        when multiple overlapping targets are detected.
        Unlike class-specific NMS, class-agnostic NMS does not consider the category of
        the target, but instead processes all categories of detection results together.
        This approach is simpler when dealing with multi-category object detection problems,
        but may result in conflicts between different categories.
        "multi_label": multiple labels per box (adds 0.5ms/img)
        "visual": whether visualization
        "strides": It indicates how many times the output of the convolutional layer has been
        resized relative to the input. It is generally a multiple of 2 and defaults to 8, 16, 32, etc.
        "anchors":  The configuration of anchors in the YOLO series. This value is related to the model.
        "names": The category names of the model, defaulting to the 80 category names of the COCO dataset.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    '''
    output = []
    for key, value in result.items():
        bs, _, ny, nx = value.shape
        out = np.reshape(value, (bs, na, no, ny, nx))
        out = np.transpose(out, (0, 1, 3, 4, 2))

        grid, anchor_grid = make_grid(anchors[key], strides[key], nx, ny)
        out = sigmoid(out)
        out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * strides[key]  # xy
        out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor_grid
        
        out = np.reshape(out, (1, -1, 85))
        output = np.concatenate(
            [output, out], axis=1) if len(output) > 0 else out

    # NMS
    pred = non_max_suppression_yolo(
        output, conf_thres, iou_thres, agnostic_nms, multi_label, max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], img0.shape[:2], ratio_pad).round()

    if visual:
        visualization(det, img0, names)
    return det


# SSD series
def post_process_ssd(result, img, img0, conf_thres=0.25, iou_thres=0.45, topk=200,
                     max_det=100, names=['ship'], scales=[27.0, 55.0, 111.0],
                     aspect_ratios=[1, 0.5, 2], visual=False, 
                     anchors_file='anchors.npy'
                     ):
    '''
    Runs post_process on inference results

    Inputs:
        "result":   The output of net.forward(), if len(result) > 1, you need to organize
        the result into a dictionary according to the output name of the layer.
        "img": Input original image.
        "img0": Input original image.
        "conf_thres": Confidence threshold, which is 0.25 by default.
        "iou_thres": NMS IoU threshold, which is 0.45 by default.
        "topk": Only process the top-k results for each image 
        "max_det": Maximum detections per image, which is 1000 by default.
        "names": The category names of the model.
        "scales": Generate scales corresponding to anchors.
        "aspect_ratios": The aspect ratio of anchors generated by the SSD series model.
        "visual": whether visualization
        "anchors_file":  The configuration of anchors in the SSD series.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    '''
    output = []
    batch, _, img_h, img_w = img.shape
    index = 0
    half = int(len(result)/2)
    conf_list = []
    loc_list = []
    for key, value in result.items():
        if index < half:
            conf_list.append(np.transpose(
                value, (0, 2, 3, 1)).reshape(batch, -1, 2))
        else:
            loc_list.append(np.transpose(
                value, (0, 2, 3, 1)).reshape(batch, -1, 4))
        index += 1

    class_pred = np.hstack(conf_list)
    box_pred = np.hstack(loc_list)
    class_pred[0] = softmax(class_pred[0])

    # anchors
    anchors = []
    shapes = [value.shape for key, value in result.items()][:half]
    if isinstance(anchors, list):
        for i, shape in enumerate(shapes):
            anchors += make_anchors(img_h, img_w, shape[2],
                                    shape[3], scales[i], aspect_ratios)
    anchors = np.array(anchors).reshape(-1, 4)
    np.save(anchors_file, anchors)

    # NMS
    pred = non_max_suppression_ssd(class_pred, box_pred, anchors,
                                   conf_thres, iou_thres, topk, max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, [0, 2]] *= img_w
            det[:, [1, 3]] *= img_h
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape).round()

    if visual:
        visualization(det, img0, names)

    return det


def post_process(*args, **kwargs):
    det = []
    if len(args) == 15 and isinstance(args[7], int):
        det = post_process_yolo(*args, **kwargs)
    elif len(args) == 12 and isinstance(args[7], list):
        det = post_process_ssd(*args, **kwargs)
    return det
