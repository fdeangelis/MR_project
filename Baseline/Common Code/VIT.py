import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch
from transformers import ViTModel, ViTConfig

'''
    Adapts a standard pretrained Vision Transformer (ViT) to classify Region of Interest (ROI) data
    where the input contains extra contextual channels beyond standard RGB.
'''

class ROIClassifierViT(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
        
        # Channel reduction layers - same as ResNet version
        self.first = nn.Conv2d(5, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Vision Transformer backbone - using ViT-Base pretrained
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Classification head - ViT outputs 768-dim features
        self.fc = nn.Linear(768, num_hoi_classes)

        
        
    def forward(self, x):
        # Channel reduction: 5 -> 4 -> 3 channels
        x = self.first(x)
        x = self.pre_conv(x)
        
        # ViT expects inputs in range [0, 1] and specific format
        # Extract features using ViT backbone
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        
        # Classification
        out = F.sigmoid(self.fc(features))
        return out

