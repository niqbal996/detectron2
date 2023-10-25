# from ..common.optim import SGD as optimizer
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
from ..common.data.sugarbeets import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

optimizer.lr = 1e-4
dataloader.train.mapper.use_instance_mask = False

# dataloader.train.total_batch_size = 30  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 30
# dataloader.test.batch_size = 20
# dataloader.test.num_workers = 20

dataloader.train.total_batch_size = 4  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 1
dataloader.test.batch_size = 4
dataloader.test.num_workers = 1

# model.backbone.bottom_up.freeze_at = 2
model.roi_heads.num_classes = 2
del model.roi_heads.mask_in_features
del model.roi_heads.mask_pooler
del model.roi_heads.mask_head

train.max_iter = 15000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)
train.output_dir = '/mnt/d/trainers/phenobench_test/frcnn/'

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

def register_dataset():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("pheno_train", {},
                            "/mnt/d/datasets/PhenoBench/coco_anns/plants_panoptic_train.json",
                            "/mnt/d/datasets/PhenoBench/train/")
    register_coco_instances("pheno_val", {},
                            "/mnt/d/datasets/PhenoBench/coco_anns/plants_panoptic_val.json",
                            "/mnt/d/datasets/PhenoBench/val/")

register_dataset()