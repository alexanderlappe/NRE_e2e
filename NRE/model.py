import torch
from backbone.EfficientFace import *
from NRE.modules import *

backbone = efficient_face()

print(backbone)

class NREModel(torch.nn.Module):
    """
    Wrapper that makes successive calls to backbone and NRE modules
    """
    def __init__(self, backbone=efficient_face(), n_classes=7, n_domains=1, in_features=15):
        super().__init__()
        self.backbone = backbone
        self.domain = DomainHead(n_domains, in_features)
        self.relative = RelativeEncoding(n_domains, in_features)
        self.projection = NREProjection(n_classes, in_features)
        self.classifier = NREClassifier()

    def forward(self, x):
        print(x.shape)
        features = self.backbone(x)
        domain_prob =self.domain(features)
        v = self.relative(features, domain_prob)
        activity = self.projection(v)
        p = self.classifier(activity)

        return p, domain_prob