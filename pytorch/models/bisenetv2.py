import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthwiseConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x = self.bn(self.dw(x))
        x = self.pw(x)
        return x


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class StemBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = ConvBNReLU(in_dim, out_dim, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(out_dim, out_dim // 2, 1),
            ConvBNReLU(out_dim // 2, out_dim, 3, stride=2)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_dim * 2, out_dim, 3)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        concat = torch.cat([left, right], dim=1)
        return self.fuse(concat)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_dim)
        self.conv_1x1 = ConvBNReLU(in_dim, out_dim, 1)
        self.add = Add()
        self.conv_3x3 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        gap = self.gap(x)
        bn = self.bn(gap)
        conv1 = self.add(self.conv_1x1(bn), x)
        return self.conv_3x3(conv1)


class GatherAndExpansionLayer1(nn.Module):
    def __init__(self, in_dim, out_dim, expand):
        super().__init__()
        expand_dim = expand * in_dim
        self.conv = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3),
            DepthwiseConvBN(in_dim, expand_dim, 3),
            ConvBN(expand_dim, out_dim, 1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class GatherAndExpansionLayer2(nn.Module):
    def __init__(self, in_dim, out_dim, expand):
        super().__init__()
        expand_dim = expand * in_dim
        self.branch_1 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3),
            DepthwiseConvBN(in_dim, expand_dim, 3, stride=2),
            DepthwiseConvBN(expand_dim, expand_dim, 3),
            ConvBN(expand_dim, out_dim, 1)
        )
        self.branch_2 = nn.Sequential(
            DepthwiseConvBN(in_dim, in_dim, 3, stride=2),
            ConvBN(in_dim, out_dim, 1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch_1(x) + self.branch_2(x))


class DetailBranch(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        C1, C2, C3 = feature_channels
        self.convs = nn.Sequential(
            ConvBNReLU(in_channels, C1, 3, stride=2),
            ConvBNReLU(C1, C1, 3),
            ConvBNReLU(C1, C2, 3, stride=2),
            ConvBNReLU(C2, C2, 3),
            ConvBNReLU(C2, C2, 3),
            ConvBNReLU(C2, C3, 3, stride=2),
            ConvBNReLU(C3, C3, 3),
            ConvBNReLU(C3, C3, 3)
        )

    def forward(self, x):
        return self.convs(x)


class SemanticBranch(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        C1, C3, C4, C5 = feature_channels
        self.stem = StemBlock(in_channels, C1)
        self.stage3 = nn.Sequential(
            GatherAndExpansionLayer2(C1, C3, 6),
            GatherAndExpansionLayer1(C3, C3, 6)
        )
        self.stage4 = nn.Sequential(
            GatherAndExpansionLayer2(C3, C4, 6),
            GatherAndExpansionLayer1(C4, C4, 6)
        )
        self.stage5_4 = nn.Sequential(
            GatherAndExpansionLayer2(C4, C5, 6),
            GatherAndExpansionLayer1(C5, C5, 6),
            GatherAndExpansionLayer1(C5, C5, 6),
            GatherAndExpansionLayer1(C5, C5, 6)
        )
        self.ce = ContextEmbeddingBlock(C5, C5)

    def forward(self, x):
        stage2 = self.stem(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.ce(stage5_4)
        return stage2, stage3, stage4, stage5_4, fm


class BGA(nn.Module):
    def __init__(self, out_dim, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.db_branch_keep = nn.Sequential(
            DepthwiseConvBN(out_dim, out_dim, 3),
            nn.Conv2d(out_dim, out_dim, 1)
        )
        self.db_branch_down = nn.Sequential(
            ConvBN(out_dim, out_dim, 3, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.sb_branch_keep = nn.Sequential(
            DepthwiseConvBN(out_dim, out_dim, 3),
            nn.Conv2d(out_dim, out_dim, 1)
        )
        self.sb_branch_up = ConvBN(out_dim, out_dim, 3)
        self.conv = ConvBN(out_dim, out_dim, 3)

    def forward(self, dfm, sfm):
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = torch.sigmoid(self.sb_branch_keep(sfm))
        sb_feat_up = self.sb_branch_up(sfm)
        sb_feat_up = F.interpolate(sb_feat_up, size=db_feat_keep.shape[2:], mode='bilinear', align_corners=self.align_corners)
        sb_feat_up = torch.sigmoid(sb_feat_up)
        db_feat = db_feat_keep * sb_feat_up
        sb_feat = db_feat_down * sb_feat_keep
        sb_feat = F.interpolate(sb_feat, size=db_feat.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return self.conv(db_feat + sb_feat)


class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes):
        super().__init__()
        self.conv_3x3 = nn.Sequential(ConvBNReLU(in_dim, mid_dim, 3), nn.Dropout(0.1))
        self.conv_1x1 = nn.Conv2d(mid_dim, num_classes, 1, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2


class BiSeNetV2(nn.Module):
    def __init__(self, num_classes=2, lambd=0.25, align_corners=False, in_channels=3, pretrained=None):
        super().__init__()
        C1, C2, C3 = 64, 64, 128
        db_channels = (C1, C2, C3)
        C1s, C3s, C4s, C5s = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1s, C3s, C4s, C5s)
        mid_channels = 128
        self.db = DetailBranch(in_channels, db_channels)
        self.sb = SemanticBranch(in_channels, sb_channels)
        self.bga = BGA(mid_channels, align_corners)
        self.aux_head1 = SegHead(C1s, C1s, num_classes)
        self.aux_head2 = SegHead(C3s, C3s, num_classes)
        self.aux_head3 = SegHead(C4s, C4s, num_classes)
        self.aux_head4 = SegHead(C5s, C5s, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)
        self.align_corners = align_corners
        self.pretrained = pretrained

    def forward(self, x):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit = self.head(self.bga(dfm, sfm))
        
        if self.training:
            logit1 = self.aux_head1(feat1)
            logit2 = self.aux_head2(feat2)
            logit3 = self.aux_head3(feat3)
            logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3, logit4]
        else:
            logit_list = [logit]

        logit_list = [
            F.interpolate(logit, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            for logit in logit_list
        ]
        
        if self.training:
            return logit_list
        else:
            return logit_list[0]
