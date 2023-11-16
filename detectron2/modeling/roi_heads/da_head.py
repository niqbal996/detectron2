# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.layers import GradientScalarLayer
from detectron2.modeling.poolers import LevelMapper
from .roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures.bounding_box import BoxList
# from detectron2.utils.registry import Registry

# __all__ = ["DAImgHead", "DAInsHead", "DomainAdaptationModule", "DA_HEAD_REGISTRY"]

# DA_HEAD_REGISTRY = Registry("DA_HEAD_REGISTRY")
# DA_HEAD_REGISTRY.__doc__ = """
# A domain adaptation head in addition to the ROI head that takes input from feature map of ResNet and FRCNN Head

# TODO: More documentation needed. 
# """
from .loss import make_da_heads_loss_evaluator

@ROI_HEADS_REGISTRY.register()
class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.da_img_conv1_layers = []
        self.da_img_conv2_layers = []
        for idx in range(5):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv2_block = "da_img_conv2_level{}".format(idx)
            conv1_block_module = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
            conv2_block_module = nn.Conv2d(512, 1, kernel_size=1, stride=1)
            for module in [conv1_block_module, conv2_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.add_module(conv2_block, conv2_block_module)
            self.da_img_conv1_layers.append(conv1_block)
            self.da_img_conv2_layers.append(conv2_block)


    def forward(self, x):
        img_features = []

        for feature, conv1_block, conv2_block in zip(
                x, self.da_img_conv1_layers, self.da_img_conv2_layers
        ):
            inner_lateral = getattr(self, conv1_block)(feature)
            last_inner = F.relu(inner_lateral)
            img_features.append(getattr(self, conv2_block)(last_inner))
        return img_features


class DAJointScaleHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAJointScaleHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels*5, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, da_img):

        _, _, H, W = da_img[0].shape
        up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)

        upsampled_feat = []
        for i, feat in enumerate(da_img):
            feat = da_img[i]
            upsampled_feat.append(up_sample(feat))

        upsampled_feat = torch.cat(upsampled_feat, dim=1)

        img_features = []
        t = F.relu(self.conv1_da(upsampled_feat))
        img_features.append(self.conv2_da(t))

        return img_features


class ScaleDiscriminator(nn.Module):
    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(ScaleDiscriminator, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 5, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class ScaleDiscriminatorIns(nn.Module):
    def __init__(self, in_channels):
        super(ScaleDiscriminatorIns, self).__init__()

        self.scale_score = nn.Linear(in_channels, 5)

        nn.init.normal_(self.scale_score.weight, std=0.01)
        for l in [self.scale_score]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.scale_score(x)
        return scores

@ROI_HEADS_REGISTRY.register()
class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()

        self.da_ins_fc1_layers = []
        self.da_ins_fc2_layers = []
        self.da_ins_fc3_layers = []

        for idx in range(4):
            fc1_block = "da_ins_fc1_level{}".format(idx)
            fc2_block = "da_ins_fc2_level{}".format(idx)
            fc3_block = "da_ins_fc3_level{}".format(idx)
            fc1_block_module = nn.Linear(in_channels, 1024)
            fc2_block_module = nn.Linear(1024, 1024)
            fc3_block_module = nn.Linear(1024, 1)
            for module in [fc1_block_module, fc2_block_module, fc3_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
            self.add_module(fc1_block, fc1_block_module)
            self.add_module(fc2_block, fc2_block_module)
            self.add_module(fc3_block, fc3_block_module)
            self.da_ins_fc1_layers.append(fc1_block)
            self.da_ins_fc2_layers.append(fc2_block)
            self.da_ins_fc3_layers.append(fc3_block)


    def forward(self, x, levels=None):

        dtype, device = x.dtype, x.device

        result = torch.zeros((x.shape[0], 1),
            dtype=dtype, device=device,)

        for level, (fc1_da, fc2_da, fc3_da) in \
                enumerate(zip(self.da_ins_fc1_layers,
                              self.da_ins_fc2_layers, self.da_ins_fc3_layers)):

            idx_in_level = torch.nonzero(levels == level).squeeze(1)

            if len(idx_in_level) > 0:
                xs = x[idx_in_level, :]

                xs = F.relu(getattr(self, fc1_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                xs = F.relu(getattr(self, fc2_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                result[idx_in_level] = getattr(self, fc3_da)(xs)

            return result

@ROI_HEADS_REGISTRY.register()
class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, 
                 da_image_grl_weight=0.01, 
                 da_ins_grl_weight=0.1, 
                 cos_weight=0.1, 
                 num_img_inputs=256,
                 num_ins_inputs=None,
                 scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                 loss_evaluator=None):
        super(DomainAdaptationModule, self).__init__()

        # self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        # num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.RPN.USE_FPN else res2_out_channels * stage2_relative_factor
        # self.USE_FPN = cfg.MODEL.RPN.USE_FPN
        self.USE_FPN = True                 # TODO dont hardcode
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.consit_weight = cos_weight

        self.grl_img = GradientScalarLayer(-1.0 * da_image_grl_weight)
        self.grl_ins = GradientScalarLayer(-1.0 * da_ins_grl_weight)
        self.grl_img_consist = GradientScalarLayer(self.consit_weight * da_image_grl_weight)
        self.grl_ins_consist = GradientScalarLayer(self.consit_weight * da_ins_grl_weight)

        # in_channels = num_img_inputs #cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(num_img_inputs)
        # self.loss_evaluator = make_da_heads_loss_evaluator(cfg)
        self.loss_evaluator = loss_evaluator

        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)
        self.inshead = DAInsHead(num_ins_inputs)

    def forward(self, proposals, img_features, da_ins_feature, da_ins_labels, da_proposals, targets=None):
        """
        Arguments:
            proposals (list[BoxList]): proposal boxes
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            da_ins_feature (Tensor): instance feature vectors extracted according to da_proposals
            da_ins_labels (Tensor): domain labels for instance feature vectors
            da_proposals (list[BoxList]): randomly selected proposal boxes
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if not self.USE_FPN:
            da_ins_feature = self.avgpool(da_ins_feature)
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(img_features[fea]) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        img_grl_consist_fea = [self.grl_img_consist(img_features[fea]) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        # instance alignment
        levels = self.map_levels(da_proposals)
        da_ins_features = self.inshead(ins_grl_fea, levels)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea, levels)

        da_ins_consist_features = da_ins_consist_features.sigmoid()

        # image alignment
        da_img_features = self.imghead(img_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_proposals, da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features,
                da_ins_labels, targets)

            losses = {
                "loss_da_image": da_img_loss,
                "loss_da_instance": da_ins_loss,
                "loss_da_consistency": da_consistency_loss}

            return losses

        return {}


def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []
