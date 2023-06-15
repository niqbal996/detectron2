# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import os, json, cv2, random
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.explain.gradcam import GradCam, GuidedBackpropReLU, GuidedBackpropReLUModel

year = 2016

def main():
    # register_coco_instances(f"sugar_beet_train", {}, f"/netscratch/naeem/structured_cwc/instances_train{year}.json",
    #                         f"/netscratch/naeem/structured_cwc/train/img/")
    # register_coco_instances(f"sugar_beet_valid", {}, f"/netscratch/naeem/structured_cwc/instances_valid{year}.json",
    #                         f"/netscratch/naeem/structured_cwc/valid/img/")

    register_coco_instances("sugar_beet_train", {},
                            "/home/robot/datasets/structured_cwc/instances_train2016.json",
                            "/home/robot/datasets/structured_cwc/train/img/")
    register_coco_instances("sugar_beet_valid", {},
                            "/home/robot/datasets/structured_cwc/instances_valid2016.json",
                            "/home/robot/datasets/structured_cwc/valid/img/")
    register_coco_instances("sugar_beet_test", {},
                            "/home/robot/datasets/structured_cwc/instances_test2016.json",
                            "/home/robot/datasets/structured_cwc/test/img/")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"sugar_beet_train",)
    cfg.DATASETS.TEST = (f"sugar_beet_test",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  #
    cfg.OUTPUT_DIR = '/home/robot/datasets/MRCNN_training'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=True)
    # trainer.train()

    # cfg already contains everything we've set previously. Now we changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join('/home/robot/git/detectron2/output/model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(f"sugar_beet_valid", cfg, False, output_dir="/home/robot/datasets/MRCNN_training")
    val_loader = build_detection_test_loader(cfg, f"sugar_beet_valid")
    # grad_cam = GradCam(model=trainer.model,
    #                    feature_module=trainer.model.layer4,
    #                    target_layer_names=["2"], use_cuda=True)
    # print(inference_on_dataset(trainer.model, val_loader, evaluator))
    dataset_dicts = DatasetCatalog.get(f"sugar_beet_valid")
    def get_label(rgb_path):
        data_root, file_name = os.path.split(os.path.split(rgb_path)[0])[0], os.path.split(rgb_path)[1]
        return os.path.join(data_root, 'lbl', file_name)

    c = 0

    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        lbl = cv2.imread(get_label(d["file_name"]))
        outputs = predictor(im)
        # outputs = grad_cam(im, 0)
        v = Visualizer(im[:, :, ::-1],
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = out.get_image()
        print(img.shape)
        img = Image.fromarray(np.concatenate([img[:, :, ::-1],
                                              cv2.resize(lbl, (img.shape[1], img.shape[0]),
                                                         interpolation=cv2.INTER_AREA)],
                                             axis=1))
        img.save(f"{cfg.OUTPUT_DIR}/output{c}.jpeg")
        c = c + 1

if __name__ == "__main__":
    main()
