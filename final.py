from transformers import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamModel,
    SamConfig,
    CLIPVisionConfig, 
    CLIPVisionModel,
)

import torch

import torch

class EncSAMCLIP():
    def __init__(self):
        pass

    def fuse(self, sam_enc, clip_enc):

        # Concatenate the outputs of the two image encoders.
        output = torch.cat([self.sam_enc.SamVisionEncoderOutput, self.clip_enc.BaseModelOutput], dim=1)

        linear_layer = torch.nn.Linear(output.shape[1], output.shape[1] // 2)
        output = linear_layer(output)

        return output


# SAM model
configuration = SamConfig()

sam_model = SamModel(configuration)
configuration = sam_model.config
vision_config = SamVisionConfig()
prompt_encoder_config = SamPromptEncoderConfig()
mask_decoder_config = SamMaskDecoderConfig()
config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)

# CLIP model
configuration = CLIPVisionConfig()
clip_model = CLIPVisionModel(configuration)

enc_sam_clip = EncSAMCLIP()
fused_encoders = enc_sam_clip.fuse(sam_model.vision_encoder, clip_model.vision_model.encoder)