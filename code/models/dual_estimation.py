import torch
from torch import nn
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from util import constants as C

class DualBranchUnet(Unet):

    def __init__(self,
                 main_backbone: str, 
                 s1_backbone: str,
                 s2_backbone: str, 
                 in_channels_s1: int, 
                 in_channels_s2: int,  
                 **kwargs
                 ):
        # Initialize the Unet with combined channels
        super().__init__(encoder_name=main_backbone,
                         in_channels=in_channels_s1+in_channels_s2, 
                         classes=1, 
                         decoder_channels=C.UNET_DECOD,
                         **kwargs
                         )

        # Initialize separate encoders for S1 and S2
        self.encoder_s1 = get_encoder(s1_backbone, in_channels=in_channels_s1)
        self.encoder_s2 = get_encoder(s2_backbone, in_channels=in_channels_s2)
        self.encoder = get_encoder(main_backbone, in_channels=in_channels_s1+in_channels_s2)

        # Modify the original decoder_channels to match the combined encoder channels
        combined_encoder_channels = [
            s1_ch + s2_ch for s1_ch, s2_ch in zip(self.encoder_s1.out_channels, self.encoder_s2.out_channels)
        ]

        # Initialize the modified UnetDecoder with the combined encoder channels
        self.decoder = UnetDecoder(
            encoder_channels=combined_encoder_channels,  # Adjusted to the combined channels
            decoder_channels=C.UNET_DECOD,  # Adjusted according to combined channels
            n_blocks=len(combined_encoder_channels)-1,  # Number of blocks to match the number of encoder levels
            use_batchnorm=True,
            attention_type=None,
        )
        last_decoder_channel = C.UNET_DECOD[-1]
        
        # Initialize the segmentation head with the correct number of input channels
        self.segmentation_head = SegmentationHead(
            in_channels=last_decoder_channel,  # Match this with the output of the last decoder block
            out_channels=kwargs.get('classes', 1),  # Number of classes for the output mask
            activation=kwargs.get('activation', 'identity'),  # Activation function if any
            kernel_size=3,
        )
        # ... any additional modifications ...

    def forward(self, x):
        # Override the forward method to split the input and process it through the dual encoders
        # Split the input into S1 and S2 components
        x_s1, x_s2 = torch.split(x, [C.NUM_S1,C.NUM_S2 ], dim=1)
        
        # Pass inputs through their respective encoders
        features_s1 = self.encoder_s1(x_s1)
        features_s2 = self.encoder_s2(x_s2)

        # Combine features from both encoders
        combined_features = [torch.cat((f1, f2), dim=1) for f1, f2 in zip(features_s1, features_s2)]
        # Process combined features through the Unet decoder and segmentation head
        decoder_output = self.decoder(*combined_features)
        # Since we have bypassed the original encoder, we must manually handle the skip connections if any
        output = self.segmentation_head(decoder_output)

        return output