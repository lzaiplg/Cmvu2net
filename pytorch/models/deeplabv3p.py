import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None, dilation=1):
        super().__init__()
        if padding is None:
            padding = (k // 2) * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(1, 6, 12, 18), align_corners=False, image_pooling=True):
        super().__init__()
        self.align_corners = align_corners
        branches = []
        branches.append(ConvBNReLU(in_channels, out_channels, 1))
        for r in atrous_rates[1:]:
            branches.append(ConvBNReLU(in_channels, out_channels, 3, dilation=r))
        self.branches = nn.ModuleList(branches)
        self.image_pooling = image_pooling
        if image_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_conv = ConvBNReLU(in_channels, out_channels, 1)
        concat_channels = out_channels * (len(self.branches) + (1 if image_pooling else 0))
        self.project = ConvBNReLU(concat_channels, out_channels, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        if self.image_pooling:
            gp = self.global_pool(x)
            gp = self.global_conv(gp)
            gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats.append(gp)
        y = torch.cat(feats, dim=1)
        y = self.project(y)
        return y


class Decoder(nn.Module):
    def __init__(self, num_classes, low_channels, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.low_proj = ConvBNReLU(low_channels, 48, 1)
        self.fuse = nn.Sequential(
            ConvBNReLU(48 + 256, 256, 3),
            ConvBNReLU(256, 256, 3),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, high_feat, low_feat):
        low = self.low_proj(low_feat)
        high = F.interpolate(high_feat, size=low.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x = torch.cat([low, high], dim=1)
        x = self.fuse(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, output_stride=8, pretrained=None):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        block = BasicBlock
        layers = [2, 2, 2, 2]

        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        
        self.pretrained = pretrained
        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return [l1, l4]

    def init_weight(self):
        if self.pretrained:
            self.load_state_dict(torch.load(self.pretrained))


class DeepLabV3P(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, aspp_ratios=(1, 12, 24, 36), align_corners=False, pretrained=None):
        super().__init__()
        self.backbone = ResNet18(output_stride=8)
        self.aspp = ASPPModule(512, 256, atrous_rates=aspp_ratios, align_corners=align_corners, image_pooling=True)
        self.decoder = Decoder(num_classes, low_channels=64, align_corners=align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        low, high = self.backbone(x)
        h = self.aspp(high)
        y = self.decoder(h, low)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return [y]

    def init_weight(self):
        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))
