import numpy as np
import cv2

# Resize input image while keeping aspect ratio
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), 
              auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def pre_process(img_file, img_shape=[640, 640],
                norm_mean=[103.53, 116.28, 123.68],
                norm_std=[57.1, 57.4, 58.4],
                isgray=False, resize_mode='force'):
    '''
    img_file: img path
    img_shape:(width,height)
    norm_mean: BGR mean,[103.53, 116.28, 123.68],ImageNet statistics value
    norm_std: BGR std,[57.1, 57.4, 58.4],ImageNet statistics value
    isgray:
        True:Reading gray images
        False:Reading RGB images
    resize_mode:
        'none':Do not resize
        'force':Force resize input image according to img_shape
        'smart':Resize input image while keeping aspect ratio
    '''

    if isgray:
        img0 = cv2.imread(img_file, 0)
        mean = np.array(norm_mean[-1], dtype=np.float32)
        std = np.array(norm_std[-1], dtype=np.float32)
    else:
        img0 = cv2.imread(img_file)
        mean = np.array(norm_mean, dtype=np.float32)
        std = np.array(norm_std, dtype=np.float32)

    # image resize
    width = img_shape[0]
    height = img_shape[1]
    if resize_mode is 'none':
        # don't resize
        img = img0
    elif resize_mode is 'force':
        # Force resize input image according to img_shape
        img = cv2.resize(img0, (width, height))
    elif resize_mode is 'smart':
        # Resize input image while keeping aspect ratio
        img = letterbox(img0, new_shape=img_shape)[0]

    # normalization
    img = img.astype(np.float32)
    img = (img - mean) / std

    # Convert
    if isgray:
        img = img[:, :, np.newaxis]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img, img0
