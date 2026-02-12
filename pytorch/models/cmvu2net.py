import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=True)
class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, padding=1)
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, padding=1)
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, padding=2, dilation=2)
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, padding=1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, padding=1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, padding=1)
    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin
class CartesianEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        c_mid = out_ch // 3
        c_last = out_ch - 2 * c_mid
        self.proj_t = nn.Conv2d(in_ch, c_mid, 1, stride=stride, bias=False)
        self.proj_f = nn.Conv2d(in_ch, c_mid, 1, stride=stride, bias=False)
        self.proj_l = nn.Conv2d(in_ch, c_last, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        t = self.proj_t(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        f = self.proj_f(x)
        l = self.proj_l(torch.flip(x, dims=[2]))
        fused = self.act(self.bn(torch.cat([t, f, l], dim=1)))
        return self.refine(fused)
class NestedAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn1 = self.ca1(x)
        x = x * attn1
        attn2 = self.ca2(x)
        x = x * attn2
        return x
class cmvu2net(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, feat_ch=32, mid_ch=16, pretrained=None):
        super().__init__()
        self.stage1 = CartesianEncoder(in_channels, feat_ch, stride=2)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = CartesianEncoder(feat_ch, feat_ch)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = CartesianEncoder(feat_ch, feat_ch)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = CartesianEncoder(feat_ch, feat_ch)
        self.stage3d = RSU4(feat_ch * 2, mid_ch, feat_ch)
        self.attn3d = NestedAttention(feat_ch)
        self.stage2d = RSU4(feat_ch * 2, mid_ch, feat_ch)
        self.attn2d = NestedAttention(feat_ch)
        self.stage1d = RSU4(feat_ch * 2, mid_ch, feat_ch)
        self.attn1d = NestedAttention(feat_ch)
        self.side1 = nn.Conv2d(feat_ch, num_classes, 3, padding=1)
        self.side2 = nn.Conv2d(feat_ch, num_classes, 3, padding=1)
        self.side3 = nn.Conv2d(feat_ch, num_classes, 3, padding=1)
        self.side4 = nn.Conv2d(feat_ch, num_classes, 3, padding=1)
        self.outconv = nn.Conv2d(4 * num_classes, num_classes, 1)
    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx4up = _upsample_like(hx4, hx3)
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        hx3d = self.attn3d(hx3d)
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2d = self.attn2d(hx2d)
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.attn1d(hx1d)
        hx1d = F.interpolate(hx1d, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        return d0, d1, d2, d3, d4
