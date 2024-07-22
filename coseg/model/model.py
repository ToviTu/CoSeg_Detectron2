from turtle import mode
from typing import final
from numpy import dtype
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.structures import ImageList
import torch.nn.functional as F
from detectron2.modeling.postprocessing import sem_seg_postprocess
from torch import nn
import torch
import numpy as np
import warnings
from safetensors.torch import load_file

from transformers import CLIPProcessor
from .seg_model import CLIPSegDecoder
from .lang_model import CLIPLang_xatten, CLIPLang


class CoSeg(nn.Module):
    def __init__(self, d_reduce=64, nhead=4, nencoder=2, ndecoder=2, lang_model="prefix"):
        super().__init__()

        if lang_model == "prefix":
            self.encoders = CLIPLang_prefix(nhead=nhead, nencoder=nencoder+ndecoder)
        else:
            self.encoders = CLIPLang_xatten(nhead=nhead, nencoder=nencoder, ndecoder=ndecoder)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.decoder = CLIPSegDecoder(d_reduce=d_reduce, nhead=nhead)

    def forward(self, pixel_values):
        return self.decoder(*self.encoders(pixel_values, output_hidden_states=True))

@META_ARCH_REGISTRY.register()
class CoSeg_wrapper(nn.Module):
    '''
    A wrapper for CoSeg to be used for evaluation
    '''

    @configurable
    def __init__(
        self,
        pixel_mean,
        pixel_std,
        size_divisibility,
        ignore_value,
        d_reduce,
        nhead,
        nencoder,
        ndecoder,
        lang_model,
        alpha,
        beta,
        temp,
        finetune_clip
    ):
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
        self.size_divisibility = size_divisibility
        self.resolution = (224, 224)
        self.ignore_value = ignore_value
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

        super().__init__()
        self.predictor = CoSeg(
            d_reduce=d_reduce, 
            nhead=nhead, 
            nencoder=nencoder, 
            ndecoder=ndecoder, 
            lang_model=lang_model
        )
        warnings.warn("Model initialized...but no registered vocabulary! ")

        if finetune_clip:
            self.predictor.encoders.vision_model.requires_grad_(True)
            
    @classmethod
    def from_config(self, cfg):
        return {
            "pixel_mean" : cfg.MODEL.PIXEL_MEAN,
            "pixel_std" : cfg.MODEL.PIXEL_STD,
            "size_divisibility" : cfg.INPUT.SIZE_DIVISIBILITY,
            "ignore_value": cfg.INPUT.IGNORE_VALUE,
            "d_reduce" : cfg.MODEL.D_REDUCE,
            "nhead" : cfg.MODEL.N_HEAD,
            "nencoder" : cfg.MODEL.N_ENCODER,
            "ndecoder" : cfg.MODEL.N_DECODER,
            "lang_model" : cfg.MODEL.LANG_MODEL,
            "alpha" : cfg.MODEL.ALPHA,
            "beta" : cfg.MODEL.BETA,
            "temp" : cfg.MODEL.TEMPERATURE,
            "finetune_clip": cfg.SOLVER.FINETUNE_CLIP
        }

    @property
    def device(self):
        return self.predictor.decoder.reduces[0].weight.device
    
    def criterion(self, mask_logits, label_logits, masks, ids):
        
        def dice_loss(y_true, y_pred):
            numerator = 2 * torch.sum(y_true * y_pred)
            denominator = torch.sum(y_true + y_pred)
            return 1 - numerator / denominator

        mask_loss_1 = nn.BCEWithLogitsLoss()
        mask_loss_2 = dice_loss
        lang_loss = nn.CrossEntropyLoss()

        l1 = mask_loss_1(mask_logits, masks)
        l2 = self.beta * mask_loss_2(masks, F.sigmoid(mask_logits))
        l3 = self.alpha * lang_loss(label_logits.permute(0, 2, 1), ids)

        return l1, l2, l3
    
    def register_vocabulary(self, labels):
        warnings.warn("Overwriting registered vocabulary")
        lang_model = CLIPLang().eval()
        processor = CLIPProcessor.from_pretrained(lang_model.clip_version)
        inputs = processor(labels, padding=True, return_tensors='pt')
        with torch.no_grad():
            label_embeddings = lang_model(**inputs)['text_embeds']
        label_embeddings.requires_grad_(False)
        label_embeddings = F.normalize(label_embeddings, dim=-1)
        del lang_model
        self.label_embeddings = label_embeddings.to(self.device)
        # Assume the last index is the unlabelled index
        self.unlabel_idx = self.label_embeddings.shape[0]-1
    
    def load_safetensor(self, path):
        self.predictor.load_state_dict(load_file(path))
    
    @torch.no_grad()
    def inference(self, pixel_values, output_size):
        # Extract Pred
        mask_logits, pred_embeddings = self.predictor(pixel_values) # (B, L, W, H) (B, L, D)
        # Prepare Pred
        predicted_embeddings = F.normalize(pred_embeddings, dim=-1).squeeze(0) # (B, L, D)
        all_class_probabilities = torch.softmax(predicted_embeddings @ self.label_embeddings.T / self.temp, dim=-1) # (B, L, C)
        predicted_ids = all_class_probabilities.argmax(dim=-1).cpu().numpy() # (B, L)

        # Think further about this
        # Maybe output logits for all classes?
        class_probabilities = all_class_probabilities.max(dim=-1).values # (B, L)
        mask_probabilities = F.sigmoid(mask_logits) # (B, L, W, H)
        mask_probabilities = F.interpolate(
            mask_probabilities, size=output_size, mode='bilinear', align_corners=False
        )
        intermediate_mask = torch.argmax(mask_probabilities * class_probabilities.view(-1, 20, 1, 1), dim=1).cpu() # (B, W, H) 

        final_mask = np.zeros(intermediate_mask.shape)
        for id in predicted_ids:
            candidate_indices = np.where(predicted_ids == id)[0]
            if len(candidate_indices) == 0 or id == self.unlabel_idx:
                continue

            candidate_probabilities = class_probabilities[candidate_indices].cpu().numpy()
            final_idx = np.argmax(candidate_probabilities)
            final_mask[intermediate_mask == candidate_indices[final_idx]] = id

        final_mask = torch.tensor(final_mask, dtype=torch.int64).squeeze(0)
        # To probability
        final_mask = F.one_hot(final_mask, num_classes=self.label_embeddings.shape[0]).permute(2, 0, 1)
        
        return [{"sem_seg": final_mask, "raw": mask_probabilities, 'input': pixel_values, "ids": predicted_ids}]
    
    def prepare_ground_truth(self, batched_inputs):
        # Ground Truth
        ids = [input['instances'].get_fields()['gt_classes'][:20] for input in batched_inputs]
        ids = torch.stack([F.pad(id, (0, 20-id.shape[0]), "constant", self.ignore_value) for id in ids])
        ids[ids > self.ignore_value] = self.ignore_value
        ids = ids.to(self.device, dtype=torch.long)

        masks = [input['instances'].get_fields()['gt_masks'][:20] for input in batched_inputs]
        masks = torch.stack([torch.concat([mask, torch.full((20-mask.shape[0], *mask.shape[1:]), 0, dtype=bool)]) for mask in masks]) # B L W H
        masks = masks.to(self.device, dtype=torch.float32)

        return masks, ids
    
    def prepare_inputs(self, batched_inputs, use_processor=False):
        images = [input['image'].to(dtype=torch.float32) for input in batched_inputs]

        if use_processor:
            return self.predictor.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)

        # Normalization
        pixels = [(image / 255 - self.pixel_mean.to(image.device)) / self.pixel_std.to(image.device) for image in images]
        pixels = ImageList.from_tensors(pixels, self.size_divisibility)

        # Resizing
        pixels_resized = F.interpolate(
            pixels.tensor, size=self.resolution, mode='bilinear', align_corners=False
        )
        return pixels_resized.to(self.device,)


    def forward(self, batched_inputs):
        pixels_resized = self.prepare_inputs(batched_inputs)        
        
        if not self.training:
            return self.inference(pixels_resized, (batched_inputs[0]['height'], batched_inputs[0]['width']))
        
        masks, ids = self.prepare_ground_truth(batched_inputs)

        # Prediction
        mask_logits, pred_embeddings = self.predictor(pixels_resized)
        pred_embeddings = F.normalize(pred_embeddings, dim=-1)
        label_logits = pred_embeddings @ self.label_embeddings.T / self.temp
        mask_logits = F.interpolate(mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # Loss
        l1, l2, l3 = self.criterion(mask_logits, label_logits, masks, ids)

        loss = {"Mask loss": l1, "Dice loss": l2, "Lang loss": l3}

        return loss
        


