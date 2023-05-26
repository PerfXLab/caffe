# encoding=utf8
'''
Detection with YOLOv5s
In this example, we will load a YOLOv5s model and use it to detect objects.

usage example:
  cd caffe
  python examples/yolov5/detect.py --scaleup --auto --visualization
'''
import os
import sys
import argparse
import ast

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

sys.path.append('./examples')
from pre_process import pre_process
from post_process import post_process

class CaffeDetection:
    def __init__(self, model_def, model_weights, image_resize,
                 output_list, norm_mean, norm_std, isgray, resize_mode,
                 rect, scaleup, auto, conf_thres, iou_thres, max_det, 
                 na, no, agnostic_nms, multi_label, visualization, strides, 
                 anchors, names):
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
        self.visualization = visualization
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
        img, img0, _, _  = pre_process(image_file, self.image_resize,
                                self.norm_mean, self.norm_std,
                                self.isgray, self.resize_mode,
                                self.scaleup, self.auto)
        ratio_pad=None
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
                           self.multi_label, self.visualization, strides, 
                           ratio_pad, anchors, self.names)             
        return det


def main(args):
    '''main '''
    detection = CaffeDetection(args.model_def, args.model_weights,
                               args.image_resize, args.output_list,
                               args.norm_mean, args.norm_std, args.isgray,
                               args.resize_mode, args.rect, args.scaleup, 
                               args.auto, args.conf_thres, args.iou_thres, 
                               args.max_det, args.na, args.no, args.agnostic_nms, 
                               args.multi_label, args.visualization, args.strides, 
                               args.anchors, args.names)
    det = detection.detect(args.image_file)


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
    parser.add_argument('--visualization', action='store_true',
                        help='whether visualization')
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


if __name__ == '__main__':
    main(parse_args())
