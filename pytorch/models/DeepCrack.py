import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCrack(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(DeepCrack, self).__init__()
        self.num_classes = num_classes

        self.enc1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.enc1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.enc3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.enc3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.enc4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.enc5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.enc5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dbn5_1 = nn.BatchNorm2d(512)
        self.drelu5_1 = nn.ReLU(inplace=True)
        self.dec5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dbn5_2 = nn.BatchNorm2d(512)
        self.drelu5_2 = nn.ReLU(inplace=True)
        self.dec5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dbn5_3 = nn.BatchNorm2d(512)
        self.drelu5_3 = nn.ReLU(inplace=True)

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dbn4_1 = nn.BatchNorm2d(512)
        self.drelu4_1 = nn.ReLU(inplace=True)
        self.dec4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dbn4_2 = nn.BatchNorm2d(512)
        self.drelu4_2 = nn.ReLU(inplace=True)
        self.dec4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dbn4_3 = nn.BatchNorm2d(256)
        self.drelu4_3 = nn.ReLU(inplace=True)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dbn3_1 = nn.BatchNorm2d(256)
        self.drelu3_1 = nn.ReLU(inplace=True)
        self.dec3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dbn3_2 = nn.BatchNorm2d(256)
        self.drelu3_2 = nn.ReLU(inplace=True)
        self.dec3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dbn3_3 = nn.BatchNorm2d(128)
        self.drelu3_3 = nn.ReLU(inplace=True)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dbn2_1 = nn.BatchNorm2d(128)
        self.drelu2_1 = nn.ReLU(inplace=True)
        self.dec2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dbn2_2 = nn.BatchNorm2d(64)
        self.drelu2_2 = nn.ReLU(inplace=True)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dbn1_1 = nn.BatchNorm2d(64)
        self.drelu1_1 = nn.ReLU(inplace=True)
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dbn1_2 = nn.BatchNorm2d(64)
        self.drelu1_2 = nn.ReLU(inplace=True)
        
        self.side1 = nn.Conv2d(64 + 64, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(128 + 64, num_classes, kernel_size=1)
        self.side3 = nn.Conv2d(256 + 128, num_classes, kernel_size=1)
        self.side4 = nn.Conv2d(512 + 256, num_classes, kernel_size=1)
        self.side5 = nn.Conv2d(512 + 512, num_classes, kernel_size=1)
        
        self.fuse = nn.Conv2d(5 * num_classes, num_classes, kernel_size=1)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.enc1_1(x)))
        enc1 = self.relu1_2(self.bn1_2(self.enc1_2(x)))
        x, idx1 = self.pool1(enc1)
        
        x = self.relu2_1(self.bn2_1(self.enc2_1(x)))
        enc2 = self.relu2_2(self.bn2_2(self.enc2_2(x)))
        x, idx2 = self.pool2(enc2)
        
        x = self.relu3_1(self.bn3_1(self.enc3_1(x)))
        x = self.relu3_2(self.bn3_2(self.enc3_2(x)))
        enc3 = self.relu3_3(self.bn3_3(self.enc3_3(x)))
        x, idx3 = self.pool3(enc3)
        
        x = self.relu4_1(self.bn4_1(self.enc4_1(x)))
        x = self.relu4_2(self.bn4_2(self.enc4_2(x)))
        enc4 = self.relu4_3(self.bn4_3(self.enc4_3(x)))
        x, idx4 = self.pool4(enc4)
        
        x = self.relu5_1(self.bn5_1(self.enc5_1(x)))
        x = self.relu5_2(self.bn5_2(self.enc5_2(x)))
        enc5 = self.relu5_3(self.bn5_3(self.enc5_3(x)))
        x, idx5 = self.pool5(enc5)
        
        x = self.unpool5(x, idx5)
        x = self.drelu5_1(self.dbn5_1(self.dec5_1(x)))
        x = self.drelu5_2(self.dbn5_2(self.dec5_2(x)))
        dec5 = self.drelu5_3(self.dbn5_3(self.dec5_3(x)))
        
        x = self.unpool4(dec5, idx4)
        x = self.drelu4_1(self.dbn4_1(self.dec4_1(x)))
        x = self.drelu4_2(self.dbn4_2(self.dec4_2(x)))
        dec4 = self.drelu4_3(self.dbn4_3(self.dec4_3(x)))
        
        x = self.unpool3(dec4, idx3)
        x = self.drelu3_1(self.dbn3_1(self.dec3_1(x)))
        x = self.drelu3_2(self.dbn3_2(self.dec3_2(x)))
        dec3 = self.drelu3_3(self.dbn3_3(self.dec3_3(x)))
        
        x = self.unpool2(dec3, idx2)
        x = self.drelu2_1(self.dbn2_1(self.dec2_1(x)))
        dec2 = self.drelu2_2(self.dbn2_2(self.dec2_2(x)))
        
        x = self.unpool1(dec2, idx1)
        x = self.drelu1_1(self.dbn1_1(self.dec1_1(x)))
        dec1 = self.drelu1_2(self.dbn1_2(self.dec1_2(x)))
        
        out1 = self.side1(torch.cat([enc1, dec1], dim=1))
        
        out2 = self.side2(torch.cat([enc2, dec2], dim=1))
        out2 = F.interpolate(out2, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        
        out3 = self.side3(torch.cat([enc3, dec3], dim=1))
        out3 = F.interpolate(out3, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        
        out4 = self.side4(torch.cat([enc4, dec4], dim=1))
        out4 = F.interpolate(out4, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        
        out5 = self.side5(torch.cat([enc5, dec5], dim=1))
        out5 = F.interpolate(out5, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        
        fused = self.fuse(torch.cat([out1, out2, out3, out4, out5], dim=1))
        
        return [fused, out1, out2, out3, out4, out5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
