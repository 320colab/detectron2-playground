from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import argparse
import cv2
import numpy as np
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('inpath', type=str, help='input path')
parser.add_argument('outpath', type=str, help='output path')
parser.add_argument('--maskclass', '-m', type=str, help='class from dataset', default=None)
parser.add_argument('--threshold', '-t', type=float, help='class from dataset', default=0.5)
parser.add_argument('--showclasses', action='store_true', help='list classes from dataset')
parser.add_argument('--resolution', '-r', type=int, nargs=2, help='width, height', default=None)
args = parser.parse_args()


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
classes_dict = {k: i for i, k in enumerate(MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]).thing_classes)}

if(args.showclasses):
    print(list(classes_dict.keys()))
    quit()

cap = cv2.VideoCapture(args.inpath)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
need_resizing=False

while(cap.isOpened()):
    flag, frame = cap.read()
    if not flag:
        break

    height, width, channels = frame.shape
    if out is None:
        if args.resolution is None:
            resolution = width, height
        else:
            resolution = tuple(args.resolution)
            need_resizing = True
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.outpath, fourcc, fps, resolution) 

    im = cv2.resize(frame, resolution) if need_resizing else frame
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    outputs = predictor(im)
    if args.maskclass:
        mask = outputs['instances'].pred_masks[outputs['instances'].pred_classes == classes_dict[args.maskclass]].cpu().numpy()
        temp = (np.stack([np.bitwise_or.reduce(mask,0)] * 3, -1) * 255).astype(np.uint8)
    else:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        temp = out_frame.get_image()[:, :, ::-1]
    out_frame = cv2.resize(temp, resolution)

    out.write(out_frame)

cap.release()
out.release()

