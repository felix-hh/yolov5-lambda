import json
import argparse
import time
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO

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
DEFAULT_SOURCE = r"C:\Users\felix\Desktop\yolov5_on_lambda\yolov5-lambda\data\images\bus.jpg"
DEFAULT_WEIGHTS = 'weights/yolov5s.pt'
DEFAULT_IMAGE_SIZE = IMAGE_SIZES[-1]
DEFAULT_DEVICE = ''
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45
DEFAULT_SAVE_CONFIG = False
DEFAULT_SAVE_ROOT = Path("/tmp/")
DEFAULT_IS_PATH = True

def base64_to_numpy(image_b64):
    img = BytesIO(base64.b64decode(image_b64))
    img = Image.open(img)
    img = np.array(img) #todo ensure that images are always converted to np.uint8
    return img

def numpy_to_b64(img): 
    img = Image.fromarray(img)
    b64 = BytesIO()
    img.save(b64, 'jpeg')
    b64 = base64.b64encode(b64.getvalue()).decode('utf-8')
    return b64


def detect(save=DEFAULT_SAVE_CONFIG, source= DEFAULT_SOURCE, weights= DEFAULT_WEIGHTS, imgsz= DEFAULT_IMAGE_SIZE,
            conf_thres= DEFAULT_CONF_THRES, iou_thres= DEFAULT_IOU_THRES, is_path= DEFAULT_IS_PATH):
    """
    Shape is HxW as typical in numpy images. 
    """
    save_txt= save # for now we either save both the image and labels or neither. 

    if is_path:
        source = Path(source)
        #save a copy because you're reversing the order of the color channels, and cv2 can't work with that without a new array
        source = cv2.imread(str(source))[:,:,::-1].copy()
    else:
        source = base64_to_numpy(source)
    
    # Directories
    save_dir = Path(increment_path(DEFAULT_SAVE_ROOT / 'runs/detection' / 'exp', exist_ok=True))  # increment run
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
        img = torch.from_numpy(np.copy(img)).to(device)
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
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    if save_txt:  # Write to file
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Add bbox to image
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

    # recover original color
    response_img = numpy_to_b64(im0)
    
    # Prepare for json response. Note that prediction contains only one element since one image is processed at a time. 
    result = {'predictions': pred[0].tolist(),
              'image': response_img}

    return result


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
