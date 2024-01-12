import os
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from PIL import Image

from semseg.util import config
from semseg.util.util import colorize
from img_stitching import stitching

cv2.ocl.setUseOpenCL(False)

input_path1 = 'input_image/test_29.jpg'
input_path2 = 'input_image/test_30.jpg'
output_folder = 'result/mask/'
config_path = 'semseg/config/project.yaml'
output_path = 'result/result.jpg'
seg_large_scale = 7


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    args = parser.parse_args()
    args.config = config_path
    args.image1 = input_path1
    args.image2 = input_path2
    args.opts = None
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.image1 = args.image1
    cfg.image2 = args.image2
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def remove_human(model, image_path, classes, mean, std, base_size, crop_h, crop_w, scales, colors):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)
    for scale in scales:
        long_size = round(scale * base_size)
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction = scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction = np.argmax(prediction, axis=2)

    # Create a copy of the original image
    mask = np.copy(image)

    # Expand the region of extracted people
    for i in range(h):
        for j in range(w):
            if prediction[i, j] == 12:
                if i-seg_large_scale > 0:
                    prediction[i-seg_large_scale, j] = 12 
                if j-seg_large_scale > 0:
                    prediction[i, j-seg_large_scale] = 12
    for i in range(h-1, -1, -1):
        for j in range(w-1, -1, -1):
            if prediction[i, j] == 12:
                if i+seg_large_scale < h:
                    prediction[i+seg_large_scale, j] = 12
                if j+seg_large_scale < w:
                    prediction[i, j+seg_large_scale] = 12

    # Set pixels corresponding to target_class to black
    mask[prediction == 12] = 0
    # Convert the NumPy array back to a PIL Image
    mask = Image.fromarray(mask)

    # gray = 類別
    # gray = np.uint8(prediction)
    # mask = colorize(gray, colors)
    image_name = image_path.split('/')[-1].split('.')[0]
    mask_path = os.path.join(output_folder, image_name + '_mask.png')
    mask.save(mask_path)
    logger.info("=> Mask saved in {}".format(mask_path))
    mask = cv2.imread(mask_path)
    return mask


def main():
    global args, logger
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info("=> creating model ...")

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    colors = np.loadtxt(args.colors_path).astype('uint8')

    if args.arch == 'psp':
        from semseg.model.pspnet import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.arch == 'psa':
        from semseg.model.psanet import PSANet
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                       shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, pretrained=False)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    img1 = remove_human(model.eval(), args.image1, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, colors)
    img2 = remove_human(model.eval(), args.image2, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, colors)
    
    img_ref = cv2.imread(input_path1)
    stitching(img1, img2, img_ref, output_path)


if __name__ == '__main__':
    main()
