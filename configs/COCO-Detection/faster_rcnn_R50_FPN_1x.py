# from ..common.optim import SGD as optimizer
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_syn_v2_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=4,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_real_v2_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=18,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


dataloader.train.mapper.use_instance_mask = False
dataloader.train.total_batch_size = 40
dataloader.train.num_workers = 30
optimizer.lr = 0.001

model.backbone.bottom_up.freeze_at = 2
# model.pixel_mean = [136.25, 137.81, 135.14]
model.roi_heads.num_classes = 2
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
del model.roi_heads.mask_head
train.max_iter = 40000

train.output_dir = '/netscratch/naeem/frcnn_r50_FPN_1x_maize_pretrained_no_freeze_synthetic_0.001'
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"