import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNN_TCN_Classifier(nn.Module):
    def __init__(self, tcn_channels=[256, 128], sequence_length=7, num_classes=1, pretrained=True):
        super(CNN_TCN_Classifier, self).__init__()
        backbone = resnet18(pretrained=pretrained)

        old_w = backbone.conv1.weight.data.clone()  # [64,3,7,7]
        backbone.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, nonlinearity='relu')
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_w                      # copia RGB
            backbone.conv1.weight[:, 3:5] = old_w.mean(dim=1, keepdim=True)  # media per depth+mask

        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))

        tcn_layers = []
        num_inputs = 512
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers += [
                nn.Conv1d(
                    in_channels=num_inputs if i == 0 else tcn_channels[i-1],
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation
                ),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
        self.tcn = nn.Sequential(*tcn_layers)
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)
        x = x.view(B, T, 512)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool1d(x)
        out = self.classifier(x)  # logit
        return out
