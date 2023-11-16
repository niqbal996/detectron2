# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, ResNet
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .fast_rcnn import FastRCNNOutputLayers
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head
from .roi_heads import select_foreground_proposals, ROIHeads, StandardROIHeads

DA_HEADS_REGISTRY = Registry("DA_HEADS_REGISTRY")
DA_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

@DA_HEADS_REGISTRY.register()
class DaROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        da_on: bool = False,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.train_on_pred_boxes = train_on_pred_boxes
        self.da_ON = da_on

    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     ret = super().from_config(cfg)
    #     ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
    #     # Subclasses that have not been updated to use from_config style construction
    #     # may have overridden _init_*_head methods. In this case, those overridden methods
    #     # will not be classmethods and we need to avoid trying to call them here.
    #     # We test for this with ismethod which only returns True for bound methods of cls.
    #     # Such subclasses will need to handle calling their overridden _init_*_head methods.
    #     if inspect.ismethod(cls._init_box_head):
    #         ret.update(cls._init_box_head(cfg, input_shape))
    #     if inspect.ismethod(cls._init_mask_head):
    #         ret.update(cls._init_mask_head(cfg, input_shape))
    #     if inspect.ismethod(cls._init_keypoint_head):
    #         ret.update(cls._init_keypoint_head(cfg, input_shape))
    #     return ret

    # @classmethod
    # def _init_box_head(cls, cfg, input_shape):
    #     # fmt: off
    #     in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
    #     pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    #     pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
    #     sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
    #     pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
    #     # fmt: on

    #     # If StandardROIHeads is applied on multiple feature maps (as in FPN),
    #     # then we share the same predictors and therefore the channel counts must be the same
    #     in_channels = [input_shape[f].channels for f in in_features]
    #     # Check all channel counts are equal
    #     assert len(set(in_channels)) == 1, in_channels
    #     in_channels = in_channels[0]

    #     box_pooler = ROIPooler(
    #         output_size=pooler_resolution,
    #         scales=pooler_scales,
    #         sampling_ratio=sampling_ratio,
    #         pooler_type=pooler_type,
    #     )
    #     # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
    #     # They are used together so the "box predictor" layers should be part of the "box head".
    #     # New subclasses of ROIHeads do not need "box predictor"s.
    #     box_head = build_box_head(
    #         cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
    #     )
    #     box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
    #     return {
    #         "box_in_features": in_features,
    #         "box_pooler": box_pooler,
    #         "box_head": box_head,
    #         "box_predictor": box_predictor,
    #     }
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        
        domain_label = torch.ones_like(targets[0]._fields['gt_classes'], dtype=torch.bool)
        targets[0]._fields['domain_labels'] = domain_label
        domain_label = torch.ones_like(targets[1]._fields['gt_classes'], dtype=torch.bool)
        targets[1]._fields['domain_labels'] = domain_label
        # Get proposals for domain adaptation
        if self.training:
            with torch.no_grad():
                da_proposals = self.label_and_sample_proposals_da(proposals, targets)
        del targets

        if self.training:
            box_features, box_proposals, detector_losses, da_ins_features, da_ins_labels, da_proposals = self._forward_box(features, proposals, da_proposals)
            # box_features, detector_losses, da_ins_features, da_ins_labels, da_proposals = self._forward_box(features, proposals, da_proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            return box_features, box_proposals, detector_losses, da_ins_features, da_ins_labels, da_proposals
        else:
            pred_instances = self._forward_box(features, proposals, da_proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], da_proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features] # filters out 4 out of 5 generated feature maps at different levels
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # del box_features  # TODO does this introduce some leaking error? 

        da_ins_features = self.box_pooler(features, [x.proposal_boxes for x in da_proposals])
        da_ins_features = self.box_head(da_ins_features)
        da_ins_predictions  = self.box_predictor(da_ins_features)   # NOTE! Here da_ins_predictions are a combination of class_logits and box_regression which can be used to generate loss value
        if self.training:
            detector_losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            
            if self.da_ON:
                detector_with_da_losses = self.box_predictor.losses(predictions, proposals)
                da_ins_labels = detector_with_da_losses['domain_mask']
                return (
                    box_features, # NOTE ??? 
                    proposals,
                    detector_losses,
                    da_ins_features,
                    da_ins_labels, 
                    da_proposals
                    )
            else:   
                return detector_losses
        else:   
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
        
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances