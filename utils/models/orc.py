from typing import Optional
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ORCFPN"]


def BNReLU(num_features, bn_type=None, **kwargs):
    return nn.Sequential(nn.BatchNorm2d(num_features, **kwargs), nn.ReLU())


class SpatialGather(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = (
            torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        )  # batch x k x c
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context, size=(h, w), mode="bilinear", align_corners=True
            )
        return context


class SpatialOCR(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        out_channels,
        scale=1,
        dropout=0.1,
        bn_type=None,
    ):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(
            in_channels, key_channels, scale, bn_type
        )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class ORCFPN(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b1",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 5,
        classes: int = 9,
        activation: Optional[str] = None,
        upsampling: int = 4,
        ocr_mid_channel: int = 256,
        orc_key_channel: int = 128,
        orc_scale: float = 1.0,
        orc_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(
                self.decoder.out_channels,
                self.decoder.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.decoder.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.decoder.out_channels,
                classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(
                self.decoder.out_channels,
                ocr_mid_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(ocr_mid_channel),
            nn.ReLU(inplace=True),
        )

        self.ocr_gather_head = SpatialGather(classes)
        self.ocr_distri_head = SpatialOCR(
            ocr_mid_channel,
            ocr_mid_channel,
            orc_key_channel,
            scale=orc_scale,
            dropout=orc_dropout,
        )

        self.classification_head = None
        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        feats = self.decoder(*features)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        masks = self.segmentation_head(feats)
        return masks
