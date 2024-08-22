import timm
import torch
from torch import nn

class vit_small_dinov2():
    
    def creat_model(nbr_classes):
        model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m',
                        pretrained=True,
                        num_classes=0,
                        img_size=224)
        model.head = torch.nn.Sequential(torch.nn.Linear(model.num_features, nbr_classes), torch.nn.Sigmoid())
        return model
    
class vit_small_eva02():
    
    def creat_model(nbr_classes):
        model = timm.create_model("eva02_tiny_patch14_224.mim_in22k",
                                  pretrained=True,
                                  num_classes=nbr_classes,
                                  img_size=224)
        # num_in_features = model.get_classifier().in_features
        # model.fc = nn.Sequential(
        # nn.BatchNorm1d(num_in_features),
        # nn.Linear(in_features=num_in_features, out_features=384, bias=False),
        # nn.ReLU(),
        # nn.BatchNorm1d(384),
        # nn.Dropout(0.4),
        # nn.Linear(in_features=384, out_features=nbr_classes, bias=False))
        return model