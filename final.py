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

class EncSAMCLIP(torch.nn.Module):
    def __init__(self, sam_encoder, clip_encoder, head_sam, head_clip):
        super().__init__()

        self.sam_encoder = sam_encoder
        self.clip_encoder = clip_encoder
        self.head_sam = head_sam
        self.head_clip = head_clip

    def forward(self, x):
        sam_features = self.sam_encoder(x)
        clip_features = self.clip_encoder.vision_projection(x.unsqueeze(1))

        sam_logits = self.head_sam(sam_features)
        clip_logits = self.head_clip(clip_features)

        features = torch.cat([sam_logits, clip_logits], dim=1)

        return features

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

head_clip = torch.nn.Linear(1024, 512)
torch.nn.init.xavier_uniform_(head_clip.weight)
torch.nn.init.constant_(head_clip.bias, 0.0)

merged_encoder = EncSAMCLIP(sam_model.vision_encoder, clip_model.vision_model.encoder, sam_model.mask_decoder, head_clip)

optimizer = torch.optim.Adam(merged_encoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

#TODO: Image to be set properly
logits = merged_encoder.forward("image")

seg_mask_pred = logits[:, :512]
clip_logits_pred = logits[:, 512:]