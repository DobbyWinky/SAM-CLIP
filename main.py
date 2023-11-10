import os
import pickle
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from torch.nn import functional as F
import cv2

class SAM_Feat:
    def __init__(self, sam_model,
               sam_model_type) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_model)
        self.sam.to(self.device)
        self.predict = SamPredictor(self.sam)
        self.currImage = None

    def read_prev_data(self, data_dir, frameID, objID):
        pObj = os.path.join(data_dir, 'objects', f'{objID:05d}', f'frame_{frameID:05d}.pkl')
        if not os.path.exists(pObj):
            return None, [], []
        with open(pObj, "rb") as fObj:
            return pickle.load(fObj)
    
    def get_features(self, mask):
        h, w = mask.shape[-2:]
        mask = (mask * 255).astype(np.uint8)
        rgb_mask = np.repeat(mask, 3, axis=-1)

        self.predict.set_image(self.currImage)

        input_mask = self.predict.transform.apply_image(rgb_mask)

        input_mask_torch = torch.as_tensor(input_mask, device=self.device)
        input_mask_torch = input_mask_torch.permute(2,0,1).contiguous()[None, :, :, :]
        mask = self.predict.model.preprocess(input_mask_torch)
        feats = self.predict.features.squeeze().permute(1, 2, 0)

        mask = F.interpolate(mask, size=feats.shape[0: 2], mode="bilinear")
        mask = mask.squeeze()[0]

        # Target feature extraction
        feats = feats[mask > 0]
        target_embedding = feats.mean(0).unsqueeze(0)
        feats = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        return feats
    
    def get_features_from_array(self, masks):
        rgb_mask = (masks.astype(np.uint8) * 255).repeat(3, axis=-1)
        feats = self.get_features(masks[::])
        print("Hello", feats.shape)
        return feats


def get_SAM_features():
    # Step0: Initialize SAM model
    model_dir = "models/segment_anything"
    sam_checkpoint = model_dir+"/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = SAM_Feat(sam_checkpoint, model_type)
    #Change this to the path where your frames are present
    # Step1: Read the masks from .pkl files
    masks = sam.read_prev_data("/Users/madhangi/Downloads/test", 1, 1)

    # Step2: Read image file
    sam.currImage = cv2.imread("/Users/madhangi/Downloads/test/frames/frame_00001.jpg")

    binary_masks = masks[0]
    # Step3: Get features sam
    sam_feat = sam.get_features_from_array(binary_masks)

    return sam_feat

def get_CLIP_features():
    # TODO

