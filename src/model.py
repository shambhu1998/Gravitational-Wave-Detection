from torch import nn 
import efficientnet_pytorch
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        n_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out 