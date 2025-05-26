import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNetModel, self).__init__()
        # تحميل النموذج مع الأوزان المدربة مسبقاً
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        
        # تعديل الطبقة الأخيرة للتصنيف الثنائي
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        
        # تجميد بعض الطبقات إذا أردت (اختياري)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            # تفكيك تجميد الطبقات الأخيرة
            for param in self.model.features[-1].parameters():
                param.requires_grad = True
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)