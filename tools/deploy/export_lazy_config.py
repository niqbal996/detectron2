#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
import cv2
from torch import Tensor, nn
import numpy as np

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import TracingAdapter, dump_torchscript_IR, scripting_with_instances
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import onnxruntime
import onnx

def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs):
    from detectron2.export import Caffe2Tracer

    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=inputs)
        return caffe2_model
    elif args.format == "onnx":
        import onnx

        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
    elif args.format == "torchscript":
        ts_model = tracer.export_torchscript()
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        # "proposal_boxes": Boxes,
        # "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        # "pred_masks": Tensor,
        # "pred_keypoints": torch.Tensor,
        # "pred_keypoint_heatmaps": torch.Tensor,
    }
    assert args.format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys
    # image = image[None, :, :, :]
    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.pt"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, export_params=True, opset_version=15, verbose=True)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format == "torchscript":
        return ts_model
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(args):

    if args.sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format='BGR')
        # Do same preprocessing as DefaultPredictor
        # aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )
        import numpy as np
        aug = T.ResizeShortestEdge(
            [600, 800], 1066
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        # cv2.imshow('img', image)
        # cv2.waitKey()
        # cv2.destroyWindow('img')
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        
        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs

def scale_boxes(boxes, current_size=(800, 1067), new_size=(1536, 2048)):
    x_factor = new_size[0] / current_size[0]
    y_factor = new_size[1] / current_size[1]
    boxes[:, 0] = boxes[:, 0] * x_factor
    boxes[:, 2] = boxes[:, 2] * x_factor
    boxes[:, 1] = boxes[:, 1] * y_factor
    boxes[:, 3] = boxes[:, 3] * y_factor
    return boxes


def infer_onnx(model_file, sample_input):
    import cv2
    exec_providers = onnxruntime.get_available_providers()
    exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in exec_providers else ['CPUExecutionProvider']

    session = onnxruntime.InferenceSession(model_file, sess_options=None, providers=exec_provider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    tmp = sample_input[0]['image'].numpy()
    tmp = np.transpose(tmp, (1,2,0)).astype(np.uint8)
    cv2.imshow('figure', tmp)
    cv2.waitKey()
    cv2.destroyWindow('figure')
    pred = session.run(None, {input_name: sample_input[0]['image'].numpy()})
    
    conf_inds = np.where(pred[2] > 0.50)
    filtered = {}
    filtered[0] = pred[0][conf_inds]
    filtered[1] = pred[1][conf_inds]
    filtered[2] = pred[2][conf_inds]
    filtered[3] = pred[3]
    filtered[0] = scale_boxes(filtered[0],
                               current_size=(sample_input[0]['image'].shape[2],
                                             sample_input[0]['image'].shape[1]),
                               new_size=(sample_input[0]['width'],
                                         sample_input[0]['height']))
    orig_image = cv2.imread('/home/niqbal/coco_test.png')
    # class_ids = {0: 'weeds', 1: 'maize'}

    for obj in range(filtered[0].shape[0]):
        box = filtered[0][obj, :]
        if filtered[1][obj] == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(orig_image,
                      pt1=(int(box[0]), int(box[1])),
                      pt2=(int(box[2]), int(box[3])),
                      color=color,
                      thickness=2)
        # cv2.putText(orig_image,
        #             '{:.2f} {}'.format(filtered[2][obj], class_ids[filtered[1][obj]]),
        #             org=(int(box[0]), int(box[1] - 10)),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5,
        #             thickness=2,
        #             color=color)
    # cv2.imwrite("./output_data/{:04}.png".format(count), orig)
    cv2.imshow('figure', orig_image)
    cv2.waitKey()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable respecialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)
    from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("maize_valid", {},
    #                         "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
    #                         "/media/naeem/T7/datasets/maize_data_coco")
    register_coco_instances("coco_2017_train_2", {},
                        "/mnt/d/datasets/coco/annotations/instances_train2017.json",
                        "/mnt/d/datasets/coco/images/train2017")
    register_coco_instances("coco_2017_val_2", {},
                        "/mnt/d/datasets/coco/annotations/instances_val2017.json",
                        "/mnt/d/datasets/coco/images/val2017")

    # cfg = setup_cfg(args)

    # create a torch model
    cfg = LazyConfig.load(args.config_file)
    # cfg.train.output_dir = "/media/naeem/T7/trainers/fcos_R_50_FPN_1x.py/output/"
    # cfg.dataloader.test.num_workers = 0  # for debugging
    # cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    torch_model = instantiate(cfg.model)
    torch_model.to(cfg.train.device)
    # model = create_ddp_model(model)
    DetectionCheckpointer(torch_model).load(cfg.train.init_checkpoint)
    torch_model.eval()

    # eval_loader = instantiate(cfg.dataloader.test)
    # for idx, inputs in enumerate(eval_loader):
    #     outputs = torch_model(inputs)
    #     image = cv2.imread(inputs[0]['file_name'])
    #     v = Visualizer(image[:, :, ::-1], scale=1.2)
    #     out = v.draw_instance_predictions(outputs[0]['instances'].to('cpu'))
    #     boxes = outputs[0]['instances']._fields['pred_boxes'].to("cpu").tensor.detach().numpy()
    #     for box_idx in range(boxes.shape[0]):
    #         out = v.draw_box(boxes[box_idx, :])
    #     cv2.imshow('fig', out.get_image())
    #     cv2.waitKey()

    # get sample data
    sample_inputs = get_sample_inputs(args)
    # eval_loader = instantiate(cfg.dataloader.test)
    # sample_inputs = next(iter(eval_loader))

    # # convert and save model
    # if args.export_method == "caffe2_tracing":
    #     exported_model = export_caffe2_tracing(cfg, torch_model, sample_inputs)
    # elif args.export_method == "scripting":
    #     exported_model = export_scripting(torch_model)
    # elif args.export_method == "tracing":
    #     exported_model = export_tracing(torch_model, sample_inputs)
    # exported_model = onnx.load('./pretrained/model.onnx')
    infer_onnx('./pretrained/model.onnx', sample_inputs)
    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        # dataset = cfg.DATASETS.TEST[0]
        # data_loader = build_detection_test_loader(cfg, dataset)

        eval_loader = instantiate(cfg.dataloader.test)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        # evaluator = COCOEvaluator(dataset, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, eval_loader, cfg.dataloader.evaluator)
        print_csv_format(metrics)
