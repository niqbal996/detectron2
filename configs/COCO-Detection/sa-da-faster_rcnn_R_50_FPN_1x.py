# from ..common.optim import SGD as optimizer
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_12ep_warmup_maize as lr_multiplier
from ..common.data.da_maize import source_dataloader, target_dataloader
from ..common.models.sa_da_faster import model
# from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

optimizer.lr = 1e-4
source_dataloader.train.mapper.use_instance_mask = False
target_dataloader.train.mapper.use_instance_mask = False

# dataloader.train.total_batch_size = 30  # 28 on 40GB and 80 on 80 GB
# dataloader.train.num_workers = 30
# dataloader.test.batch_size = 20
# dataloader.test.num_workers = 20

source_dataloader.train.total_batch_size = 1  # 28 on 40GB and 80 on 80 GB
source_dataloader.train.num_workers = 1
source_dataloader.test.batch_size = 1
source_dataloader.test.num_workers = 1

target_dataloader.train.total_batch_size = 1  # 28 on 40GB and 80 on 80 GB
target_dataloader.train.num_workers = 1
# target_dataloader.test.batch_size = 1
# target_dataloader.test.num_workers = 1


# model.backbone.bottom_up.freeze_at = 2
model.roi_heads.num_classes = 3

train.max_iter = 15000
train.eval_period = 200
train.log_period = 10
train.checkpointer = dict(period=200, max_to_keep=10)
train.output_dir = '/opt/datasets/trainers/sada_faster'

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# dataloader = source_dataloader + target_dataloader
print('hold')