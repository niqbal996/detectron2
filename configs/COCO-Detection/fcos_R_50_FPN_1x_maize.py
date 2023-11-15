from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
from detectron2.data.datasets import register_coco_instances
# from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 1e-4

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 2
# dataloader.train.total_batch_size = 70  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 50
# dataloader.test.batch_size = 60
# dataloader.test.num_workers = 40


dataloader.train.total_batch_size = 28  # 28 on 40GB and 80 on 80 GB
dataloader.train.num_workers = 28
dataloader.test.batch_size = 20
dataloader.test.num_workers = 20

train.output_dir = "/netscratch/naeem/fcos_pretrained_frozen_synthetic_1e-4"
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.max_iter = 15000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)

register_coco_instances("maize_syn_v2_train", {},
                        "/netscratch/naeem/maize_syn_v3/instances_train_2022.json",
                        "/netscratch/naeem/maize_syn_v3/data_2")
register_coco_instances("maize_real_v2_val", {},
                        "/netscratch/naeem/maize_real_all_days/coco_annotations/all_data.json",
                        "/netscratch/naeem/maize_real_all_days/data")

# register_coco_instances("maize_syn_v2_train", {},
#                         "/home/niqbal/datasets/maize_syn_v2/instances_train_2022_2.json",
#                         "/home/niqbal/datasets/maize_syn_v2/camera_main_camera/rect")
# register_coco_instances("maize_real_v2_val", {},
#                         "/home/niqbal/datasets/GIL_dataset/all_days/coco_annotations/all_data.json",
#                         "/home/niqbal/datasets/GIL_dataset/all_days/data")
