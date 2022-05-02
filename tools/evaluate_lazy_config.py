from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2


def register_dataset():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("maize_train", {},
                            "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_train.json",
                            "/media/naeem/T7/datasets/maize_data_coco")
    register_coco_instances("maize_valid", {},
                            "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
                            "/media/naeem/T7/datasets/maize_data_coco")

args = default_argument_parser().parse_args()
cfg_file = "../configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py"
cfg = LazyConfig.load(cfg_file)
cfg.train.output_dir = "/media/naeem/T7/trainers/fcos_R_50_FPN_1x.py/output/"
cfg.dataloader.test.num_workers = 0 # for debugging
# cfg = LazyConfig.apply_overrides(cfg, args.opts)
default_setup(cfg, args)
register_dataset()

model = instantiate(cfg.model)
model.to(cfg.train.device)
# model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

eval_loader = instantiate(cfg.dataloader.test)
model.eval()
for idx, inputs in enumerate(eval_loader):
    outputs = model(inputs)
    image = cv2.imread(inputs[0]['file_name'])
    v = Visualizer(image[:, :, ::-1], scale=1.2)
    out = v.draw_instance_predictions(outputs[0]['instances'].to('cpu'))
    boxes = outputs[0]['instances']._fields['pred_boxes'].to("cpu").tensor.detach().numpy()
    for box_idx in range(boxes.shape[0]):
        out = v.draw_box(boxes[box_idx, :])
    cv2.imshow('fig', out.get_image())
    cv2.waitKey()
    print('hold')