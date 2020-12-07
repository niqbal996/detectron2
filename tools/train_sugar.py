
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import os
import random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.data.datasets import register_coco_instances
register_coco_instances("sugar_beet_train", {},
                        "/home/robot/datasets/structured_cwc/instances_train2016.json",
                        "/home/robot/datasets/structured_cwc/train/img/")
register_coco_instances("sugar_beet_valid", {},
                        "/home/robot/datasets/structured_cwc/instances_valid2016.json",
                        "/home/robot/datasets/structured_cwc/valid/img/")
# register_coco_instances("sugar_beet_test", {},
#                         "/home/robot/datasets/structured_cwc/instances_test2016.json",
#                         "/home/robot/datasets/structured_cwc/test/img/")

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("sugar_beet_train")
dataset_dicts = DatasetCatalog.get("sugar_beet_valid")
import random
from detectron2.utils.visualizer import Visualizer


for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow(vis.get_image()[:, :, ::-1])
    cv2.imshow('image', vis.get_image())
    cv2.waitKey()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("sugar_beet_train",)
cfg.DATASETS.TEST = ("sugar_beet_valid",)
cfg.OUTPUT_DIR = "/home/robot/datasets/MRCNN_training"

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

cfg.SOLVER.IMG_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.TEST.EVAL_PERIOD = 500

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
