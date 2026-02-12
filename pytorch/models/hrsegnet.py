import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegBlock(nn.Module):
    def __init__(self, base=64, stage_index=1):
        super().__init__()
        self.h_conv1 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        if stage_index == 1:
            self.l_conv1 = nn.Sequential(
                nn.Conv2d(base, base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU(inplace=True),
            )
        elif stage_index == 2:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base, base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU(inplace=True),
            )
        elif stage_index == 3:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base, base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU(inplace=True),
                nn.Conv2d(base * int(math.pow(2, stage_index)), base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError("stage_index must be 1, 2 or 3")

        self.l_conv2 = nn.Sequential(
            nn.Conv2d(base * int(math.pow(2, stage_index)), base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
            nn.ReLU(inplace=True),
        )
        self.l_conv3 = nn.Sequential(
            nn.Conv2d(base * int(math.pow(2, stage_index)), base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
            nn.ReLU(inplace=True),
        )

        self.l2h_conv1 = nn.Conv2d(base * int(math.pow(2, stage_index)), base, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2h_conv2 = nn.Conv2d(base * int(math.pow(2, stage_index)), base, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2h_conv3 = nn.Conv2d(base * int(math.pow(2, stage_index)), base, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        size = x.shape[2:]
        out_h1 = self.h_conv1(x)
        out_l1 = self.l_conv1(x)
        out_l1_i = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True)
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        out_l2_i = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        out_l3_i = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3
        return out_hl3


class SegHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2d(inplanes, interplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(inplace=True),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.ConvTranspose2d(inplanes, interplanes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out


class HrSegNetB32(nn.Module):
    def __init__(self, in_channels=3, base=32, num_classes=2, pretrained=None):
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base // 2, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)
        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.head = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=False)

    def forward(self, x):
        h, w = x.shape[2:]
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)
        last_out = self.head(hrseg3_out)
        
        if self.training:
            seghead1_out = self.aux_head1(hrseg1_out)
            seghead2_out = self.aux_head2(hrseg2_out)
            logit_list = [last_out, seghead1_out, seghead2_out]
            logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
            return logit_list
        else:
            last_out = F.interpolate(last_out, size=(h, w), mode='bilinear', align_corners=True)
            return last_out
