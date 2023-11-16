"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import consistency_loss
from detectron2.modeling.matcher import Matcher
from detectron2.structures.boxlist_ops import cat_boxlist
from detectron2.modeling.poolers import Pooler
# from ..utils import cat
from .roi_heads import ROI_HEADS_REGISTRY
# from detectron2.utils.registry import Registry

# __all__ = ["DALossComputation", "DA_LOSS_REGISTRY"]

# DA_LOSS_REGISTRY = Registry("DA_LOSS_REGISTRY")
# DA_LOSS_REGISTRY.__doc__ = """
# A domain adaptation loss head in addition to the ROI head that takes input from feature map of ResNet and FRCNN Head

# TODO: More documentation needed. 
# """

@ROI_HEADS_REGISTRY.register()
class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self,
                 resolution=7,
                 scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                 sampling_ratio=2):
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        
    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            # is_source = targets_per_image.get_field('is_source')
            is_source = targets_per_image._fields['gt_classes']
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

    # def __call__(self, proposals, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets, da_img_features_joint):
    def __call__(self, proposals, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            proposals (list[BoxList])
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        # masks = self.prepare_masks(targets)
        # masks = torch.cat(masks, dim=0)
        # NOTE! This assumes that first sample is always source domain and second sample is always target domain
        masks = torch.tensor([1, 0], device=targets[0]._fields['gt_classes'].device)

        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment

        # da_img_loss = []
        # for da_img_per_level in da_img:
        #     N, A, H, W = da_img_per_level.shape
        #     da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
        #
        #     da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
        #     da_img_label_per_level[masks, :] = 1
        #
        #     da_img_per_level = da_img_per_level.reshape(N, -1)
        #     da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        #
        #     da_img_loss.append(F.binary_cross_entropy_with_logits(da_img_per_level, da_img_label_per_level)/len(da_img))
        #
        # da_img_loss = torch.sum(torch.stack(da_img_loss))

        # new da img
        _, _, H, W = da_img[0].shape
        up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
        upsampled_loss = []
        for i, feat in enumerate(da_img):
            feat = da_img[i]
            feat = up_sample(feat)
            da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1
            lv_loss = F.binary_cross_entropy_with_logits\
                (feat, da_img_label_per_level, reduction='none')
            upsampled_loss.append(lv_loss)

        da_img_loss = torch.stack(upsampled_loss)
        # da_img_loss, _ = torch.median(da_img_loss, dim=0)
        # da_img_loss, _ = torch.max(da_img_loss, dim=0)
        # da_img_loss, _ = torch.min(da_img_loss, dim=0)
        da_img_loss = da_img_loss.mean()

        #da img joint
        # feat = da_img_features_joint[0]
        # feat = up_sample(feat)
        # da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
        # da_img_label_per_level[masks, :] = 1
        # joint_loss = F.binary_cross_entropy_with_logits \
        #     (feat, da_img_label_per_level)

        #ins da
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_img_rois_probs = self.pooler(da_img_consist, proposals)
        da_img_rois_probs_pool = self.avgpool(da_img_rois_probs)
        da_img_rois_probs_pool = da_img_rois_probs_pool.view(da_img_rois_probs_pool.size(0), -1)

        # da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)
        da_consist_loss = F.l1_loss(da_img_rois_probs_pool, da_ins_consist)

        return da_img_loss, da_ins_loss, da_consist_loss

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator
