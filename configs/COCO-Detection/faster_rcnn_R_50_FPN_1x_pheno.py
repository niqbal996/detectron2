# from ..common.optim import SGD as optimizer
from detectron2.data.datasets import register_coco_instances
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
from ..common.data.sugarbeets import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

optimizer.lr = 1e-4
dataloader.train.mapper.use_instance_mask = False

dataloader.train.total_batch_size = 60  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 60
dataloader.test.batch_size = 40
dataloader.test.num_workers = 40

# dataloader.train.total_batch_size = 4  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 1
# dataloader.test.batch_size = 4
# dataloader.test.num_workers = 1

# model.backbone.bottom_up.freeze_at = 2
model.roi_heads.num_classes = 2
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
del model.roi_heads.mask_head

train.max_iter = 15000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)
train.output_dir = "/netscratch/naeem/phenobench_frcnn_syn_data_transformed"

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

register_coco_instances("pheno_train", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_train.json",
                        "/netscratch/naeem/phenobench/train/")
register_coco_instances("syn_pheno_train", {},
                        "/netscratch/naeem/sugarbeet_syn_v1/coco_annotations/instances_2023_train.json",
                        # "/netscratch/naeem/sugarbeet_syn_v1/images")
                        "/netscratch/naeem/sugarbeet_syn_v1/images_2")
register_coco_instances("pheno_val", {},
                        "/netscratch/naeem/phenobench/coco_annotations/coco_plants_panoptic_val.json",
                        "/netscratch/naeem/phenobench/val/")
