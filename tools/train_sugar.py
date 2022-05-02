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
from detectron2.config import get_cfg, LazyConfig
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model


register_coco_instances("maize_train", {},
                        "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_train.json",
                        "/media/naeem/T7/datasets/maize_data_coco")
register_coco_instances("maize_valid", {},
                        "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
                        "/media/naeem/T7/datasets/maize_data_coco")
# register_coco_instances("sugar_beet_valid", {},
#                         "/media/naeem/T7/datasets/datasets/dataset_root/beets/coco/val_annotations.json",
#                         "/media/naeem/T7/datasets/datasets/dataset_root/beets/coco/val")
# register_coco_instances("sugar_beet_test", {},
#                         "/media/naeem/T7/datasets/datasets/dataset_root/beets/coco/test_annotations.json",
#                         "/media/naeem/T7/datasets/datasets/dataset_root/beets/coco/test")

#visualize training data
# my_dataset_train_metadata = MetadataCatalog.get("maize_train")
# dataset_dicts = DatasetCatalog.get("maize_train")
import random
from detectron2.utils.visualizer import Visualizer

#
# for d in random.sample(dataset_dicts, 10):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     # cv2.imshow(vis.get_image()[:, :, ::-1])
#     cv2.imshow('image', vis.get_image())
#     cv2.waitKey()

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "/media/naeem/T7/trainers/faster_rcnn_R_50_FPN_1x.yaml"

cfg.DATASETS.TRAIN = ("maize_train",)
# cfg.DATASETS.TEST = ("sugar_beet_test",)
cfg.DATASETS.EVAL = ("maize_val")



cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS ="/media/naeem/T7/trainers/retinanet_R_101_FPN_3x.yaml/"
cfg.SOLVER = {}
cfg.SOLVER.IMG_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001
#
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 270000
cfg.SOLVER.STEPS = (210000, 250000)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.TEST.EVAL_PERIOD = 500

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('maize_val', exist_ok=True)
            output_folder = 'maize_val'
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=['bbox'],
                             distributed=False,
                             output_dir=output_folder)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
predictor = DefaultPredictor(cfg)
evaluator = trainer.build_evaluator(cfg,
                                    dataset_name='maize_val',
                                    output_folder=os.path.join(cfg.OUTPUT_DIR, 'maize_val'))
    # model = trainer.build_model(cfg)
    # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

trainer.resume_or_load(resume=True)
trainer.train()


dataset_dicts = DatasetCatalog.get("maize_valid")
input_data = []
output_data = []
# def get_all_inputs_outputs():
#   for data in dataset_dicts:
#     yield
# evaluator.reset()
# for inputs, outputs in get_all_inputs_outputs():
for data in dataset_dicts:
    image = cv2.imread(data['file_name'])
    # outputs = predictor(image)
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # # if len(outputs['instances']) != 0:
    # boxes = outputs["instances"]._fields['pred_boxes'].to("cpu").tensor.numpy()[0]
    # out = v.draw_box(boxes)
    # cv2.imshow('fig', out.get_image())
    # cv2.waitKey()
    input_data.append(data), output_data.append(predictor(cv2.imread(data['file_name'])))

import pickle
with open(os.path.join(cfg.OUTPUT_DIR,'input_data.pkl'), 'wb') as f:
    pickle.dump(input_data, f)
with open(os.path.join(cfg.OUTPUT_DIR, 'output_data.pkl'), 'wb') as f:
    pickle.dump(output_data, f)

with open(os.path.join(cfg.OUTPUT_DIR,'input_data.pkl'), 'rb') as f:
    input_data = pickle.load(f)
with open(os.path.join(cfg.OUTPUT_DIR, 'output_data.pkl'), 'rb') as f:
    output_data = pickle.load(f)
evaluator.process(input_data, output_data)
eval_results = evaluator.evaluate()
# trainer.tr