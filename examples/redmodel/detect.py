# encoding=utf8
'''
Detection with redmodel
In this example, we will load a redmodel and use it to detect objects.

usage example:
  cd caffe
  python examples/redmodel/detect.py --isgray --scaleup --auto --visualization
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

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))


class CaffeDetection:
    def __init__(self, model_def, model_weights, image_resize,
                 output_list, norm_mean, norm_std, isgray, resize_mode,
                 rect, scaleup, auto, conf_thres, iou_thres, topk, max_det, 
                 names, scales, aspect_ratios, visualization,  anchors_file):
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
        self.topk = topk
        self.max_det = max_det
        self.names = names
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.visualization = visualization
        self.anchors_file = anchors_file

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def detect(self, image_file):
        '''
        redmodel detection
        '''
        img, img0, _, _ = pre_process(image_file, self.image_resize,
                                self.norm_mean, self.norm_std,
                                self.isgray, self.resize_mode,
                                self.rect, self.scaleup, self.auto)
        shape = img.shape
        self.net.blobs['data'].reshape(
            shape[0], shape[1], shape[2], shape[3])
        self.net.blobs['data'].data[...] = img
        
        # Forward pass.
        self.net.forward()
 
        conf1 = self.output_list[0]
        conf2 = self.output_list[1]
        conf3 = self.output_list[2]
        loc1 = self.output_list[3]
        loc2 = self.output_list[4]
        loc3 = self.output_list[5]
        result = {}
        result[conf1] = self.net.blobs[conf1].data
        result[conf2] = self.net.blobs[conf2].data
        result[conf3] = self.net.blobs[conf3].data
        result[loc1] = self.net.blobs[loc1].data
        result[loc2] = self.net.blobs[loc2].data
        result[loc3] = self.net.blobs[loc3].data
        det = post_process(result, img, img0, self.conf_thres, self.iou_thres, self.topk,
                           self.max_det, self.names, self.scales, self.aspect_ratios,
                           self.visualization, self.anchors_file)
        return det


def main(args):
    '''main '''
    detection = CaffeDetection(args.model_def, args.model_weights,
                               args.image_resize, args.output_list,
                               args.norm_mean, args.norm_std, args.isgray,
                               args.resize_mode, args.rect, args.scaleup,
                               args.auto, args.conf_thres, args.iou_thres, 
                               args.topk, args.max_det, args.names, args.scales, 
                               args.aspect_ratios, args.visualization, 
                               args.anchors_file)
    det = detection.detect(args.image_file)


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def',
                        default='examples/redmodel/redmodel.prototxt', type=str,
                        help='prototxt path')
    parser.add_argument(
        '--image_resize', default=[640, 512], type=int, nargs=2,
        help='image input shape[width,height]')
    parser.add_argument('--model_weights',
                        default='examples/redmodel/redmodel.caffemodel', type=str,
                        help='caffemodel path')
    parser.add_argument('--image_file', default='examples/redmodel/10.jpg', type=str,
                        help='image path')
    parser.add_argument('--isgray', action='store_true',
                        help='whether to read single-channel images')
    parser.add_argument(
        '--norm_mean', default=[103.53, 116.28, 123.68], type=float, nargs=3, help='normalization mean')
    parser.add_argument(
        '--norm_std', default=[57.1, 57.4, 58.4], type=float, nargs=3,
        help='normalization standard deviation')
    parser.add_argument('--resize_mode', default='force', type=str,
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
        '--output_list', default=["Conv_231", "Conv_249", "Conv_267",
                                  "Conv_239", "Conv_257", "Conv_275"],
        type=str, nargs=6,
        help='output layer name')
    parser.add_argument('--na', default=3, type=int, help='the number \
                        of anchors corresponding to each pixel point')
    parser.add_argument('--no', default=85, type=int, help='the number of categories \
        + 5 (positive sample probability and bounding box coordinates)')
    parser.add_argument(
        '--anchors_file', default='examples/redmodel/anchors.npy',
        type=str, help='anchor file')
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
    parser.add_argument('--scales', default=[27.0, 55.0, 111.0], type=float, nargs=3,
        help='scale parameter of anchors')
    parser.add_argument('--aspect_ratios', default=[1, 0.5, 2], type=float, nargs=3,
        help='aspect ratio parameter of anchors')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--topk', default = 200, type=int, 
                        help='Only process the top-k results for each image')
    parser.add_argument('--max_det', default=100, type=int,
                        help='maximum detections per image')
    parser.add_argument('--agnostic_nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--multi_label', action='store_true',
                        help='multiple labels per box (adds 0.5ms/img)')
    parser.add_argument('--visualization', action='store_true',
                        help='whether visualization')
    parser.add_argument('--names', default=['ship'], nargs='+', type=str,
                        help='category names')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
