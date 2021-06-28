import json
import argparse
import time
from pathlib import Path
import base64
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImageArray
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

IMAGE_SIZES = (320, 640)
DEFAULT_SOURCE = 'data/images/assistant-surgeon-holds-surgical-instruments-medical-tray-78997883.jpg'
DEFAULT_WEIGHTS = 'weights/granular.pt'
DEFAULT_IMAGE_SIZE = IMAGE_SIZES[-1]
DEFAULT_DEVICE = ''
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45

def base64_to_numpy(img, shape):
    img = base64.decodebytes(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = img.reshape(shape)
    return img


def detect(save=True, source= DEFAULT_SOURCE, weights= DEFAULT_WEIGHTS, imgsz= DEFAULT_IMAGE_SIZE,
            conf_thres= DEFAULT_CONF_THRES, iou_thres= DEFAULT_IOU_THRES, is_path= True, shape= (640, 480)):
    """
    Shape is HxW as typical in numpy images. 
    """
    save_txt= save # for now we either save both the image and labels or neither. 

    if is_path:
        source = Path(source)
        source = cv2.imread(str(source))
        shape = source.shape
    else:
        source = base64_to_numpy(source, shape)
    
    # Directories
    save_dir = Path(increment_path(Path('runs/detection') / 'exp', exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImageArray(source[np.newaxis, :], img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s

            # Generate path names
            save_path_counter = 0
            while True: 
                save_path = save_dir / (str(save_path_counter) + '.jpg')
                txt_path = save_dir / 'labels' / (str(save_path_counter) + '.txt')
                if save_path in save_dir.iterdir() or ((save_dir / 'labels').exists() and txt_path in (save_dir / 'labels').iterdir()):
                    save_path_counter += 1
                    continue
                else:
                    save_path = str(save_path)
                    txt_path = str(txt_path)
                    break

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)  # label format
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            if save:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

    if save_txt or save:
        print('Results saved to %s' % save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))

    im0 = np.uint8(im0*255)
    # Prepare for json response. Note that prediction contains only one element since one image is processed at a time. 
    result = {'predictions': pred[0].tolist(),
              'image': im0.tolist()}

    return json.dumps(result)


if __name__ == '__main__':
    with torch.no_grad():
        detect()


# ipython
# %load_ext autoreload
# %autoreload 02
# from detect import *
# img = cv2.imread('data/images/bus.jpg')
# encoded = base64.b64encode(img)
# detect(source= encoded)
