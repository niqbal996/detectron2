'''
pip install opencv-contrib-python
python -m pip install -e detectron2
pip install git+git://github.com/waspinator/coco.git@2.1.0
pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
'''
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
register_coco_instances("maize_train", {},
                        "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_train.json",
                        "/media/naeem/T7/datasets/maize_data_coco")
register_coco_instances("maize_valid", {},
                        "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
                        "/media/naeem/T7/datasets/maize_data_coco")
# model = 'retinanet_R_101_FPN_3x.yaml'
model = 'faster_rcnn_R_50_FPN_1x.yaml'
# model = 'retinanet_R_50_FPN_1x.yaml'

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("maize_train")
dataset_dicts = DatasetCatalog.get("maize_valid")
# import random
# from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("../configs/COCO-Detection/{}".format(model)))
cfg.DATASETS.TRAIN = ("maize_train",)
# cfg.DATASETS.TEST = ()
cfg.DATASETS.EVAL = ("maize_valid")
cfg.OUTPUT_DIR = "/media/naeem/T7/trainers/{}".format(model)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/{}".format(model))

cfg.SOLVER.IMG_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.001

#cfg.SOLVER.WARMUP_ITERS = 1000
#cfg.SOLVER.MAX_ITER = 10000
#cfg.SOLVER.STEPS = (1500, 6000)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
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
trainer.build_evaluator(cfg, 'maize_valid', output_folder='/home/naeem/git/detectron2/output')
