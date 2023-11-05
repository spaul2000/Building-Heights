import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import constants as C
from torchsummary import summary



class SegNet(nn.Module):

    ARCHITECTURE = {
        'UNet': smp.Unet,
    }

    def __init__(self, architecture: str, backbone: str, n_input_bands: int):
        """Wrapper class for setting up segmentation architectures

            Args: 
                architecture: segmentation architecture 
                backbone: encoder type 
                n_input_bands: number of channels in x

        """
        super().__init__()

        self.model_fn = self.ARCHITECTURE[architecture]
        if architecture == "UNet":
            self.model = self.model_fn(in_channels=n_input_bands,
                                       classes=1,
                                       encoder_name=backbone,
                                       decoder_channels=C.UNET_DECOD)
            # summary(self.model, input_size=(5,128,128))

    def forward(self, x):
        return self.model(x)