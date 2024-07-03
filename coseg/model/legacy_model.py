import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
import warnings
from .lang_model import CLIPLang

class CLIPLang_xatten_legacy(nn.Module):

    '''
    This class is for detecting most salient object
    in the scene as an auto-regressive task
    '''

    def __init__(self, nhead=4, nencoder=4, ndecoder=4, clip_version="openai/clip-vit-base-patch16"):
        super().__init__()

        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_version)
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_version)

        # vision model
        self.vision_encoder = vision_model.vision_model
        self.vision_projector = vision_model.visual_projection

        # text model
        self.text_model = text_model.text_model
        self.text_projector = text_model.text_projection

        # internal dimensions
        self.d_text = self.text_model.embeddings.token_embedding.weight.shape[1]
        self.d_image = self.vision_encoder.embeddings.position_embedding.weight.shape[1]

        # Learnable embeddings
        self.query_embeddings = nn.Embedding(20, self.d_text)
        self.query_embeddings.weight.data.normal_(mean=0, std=0.02)

        # transformer for next label prediction
        self.decoder = nn.Transformer(
            d_model = self.d_text,
            nhead = nhead,
            num_encoder_layers = nencoder,
            num_decoder_layers = ndecoder,
            activation = nn.GELU(),
            batch_first = True
        )


    def to_embedding(self, input_ids):
        # index -> text_embeddings
        return self.text_model.embeddings.token_embedding(input_ids)

    def position_encode(self, embeddings):
        index_tensor = torch.arange(embeddings.shape[1]).repeat(embeddings.shape[0], 1).to(embeddings.device)
        return embeddings + self.text_model.embeddings.position_embedding(index_tensor)

    def visual_forward(self, pixel_values, output_hidden_states=False):
        # image -> text_embeddings
        return self.vision_encoder(pixel_values, output_hidden_states=output_hidden_states)

    def text_forward(self, text_embeddings):
        # text_embedding -> text_embeddings
        return self.text_model.encoder(text_embeddings)

    def decoder_forward(self, img_src, txt_tgt):
        # Send image seq and text seq to lang model
        return self.decoder(
            img_src,
            txt_tgt,
        )

    def cond_forward(self, pixel_values, output_hidden_states = False):
        # Get visual features
        visual_outputs = self.visual_forward(pixel_values, output_hidden_states=output_hidden_states)
        visual_features = self.vision_projector(visual_outputs.last_hidden_state)

        # Remove the [cls] token
        visual_features = visual_features[:, 1:, :]

        # Initialize query tokens
        index_tensor = torch.arange(20).repeat(pixel_values.shape[0], 1).to(pixel_values.device)
        query_tokens = self.query_embeddings(index_tensor)
        query_tokens = self.position_encode(query_tokens)

        # Decode texts
        text_pred = self.decoder_forward(visual_features, query_tokens)

        # Return both text embeddings and visual activations
        if output_hidden_states:
            return text_pred, [visual_outputs.hidden_states[i] for i in (3, 6, 9)]

        # Get text embeddings
        return text_pred

    def forward(self, pixel_values, output_hidden_states = False):

        if output_hidden_states:
            text_pred, hidden_states = self.cond_forward(pixel_values, output_hidden_states=output_hidden_states)
            return text_pred, hidden_states

        return self.cond_forward(pixel_values)

"""## Segmentation Model"""

class CoSeg_legacy(nn.Module):
    def __init__(self, d_reduce=64, nhead=4, nencoder=4, ndecoder=4, **kwargs):
        super().__init__()

        self.encoders = CLIPLang_xatten_legacy(nhead=nhead, nencoder=nencoder, ndecoder=ndecoder)
        self.reduces = nn.ModuleList([
            nn.Linear(self.encoders.d_image, d_reduce) for _ in range(3)
        ])
        self.film_mul = nn.Linear(self.encoders.d_text, d_reduce)
        self.film_add = nn.Linear(self.encoders.d_text, d_reduce)

        self.decoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_reduce,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation=nn.GELU(),
                    batch_first=True,
                ),
                num_layers=1,
            )
            for _ in range(3)
        ])

        self.mask_head = nn.Sequential(
            nn.Conv2d(d_reduce, d_reduce, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce, d_reduce//2, kernel_size=(4, 4), stride=(4, 4)),
            nn.GELU(),
            nn.ConvTranspose2d(d_reduce//2, 1, kernel_size=(4, 4), stride=(4, 4))
        )

    def forward(self, pixel_values):
        # Get text embeddings
        lang_output, hidden_states = self.encoders.cond_forward(pixel_values, output_hidden_states=True)
        
        # Test
        #lang_output = F.normalize(lang_output, dim=-1)

        # Image sequence size
        self.image_seq_size = int(np.sqrt(hidden_states[0].shape[1]))

        masks = []
        for i, batch_embeddings in enumerate(lang_output.permute(1, 0, 2)):
            a  = None
            for hs, block, reduce in zip(hidden_states, self.decoder, self.reduces):
                hs = hs.permute(1, 0, 2)
                if a is None:
                    a = reduce(hs)
                else:
                    a = a + reduce(hs)

                a = a * self.film_mul(batch_embeddings) + self.film_add(batch_embeddings)
                a = block(a)

            a = a[1:].permute(1, 2, 0)
            a = a.view(a.shape[0], a.shape[1], self.image_seq_size, self.image_seq_size)
            a = self.mask_head(a)
            masks.append(a)

        masks = torch.cat(masks, dim=1)
        return masks, lang_output

@META_ARCH_REGISTRY.register()
class CoSeg_legacy_wrapper(nn.Module):
    '''
    A wrapper for CoSeg_legacy to be used for evaluation
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
        self.predictor = CoSeg_legacy(
            d_reduce=d_reduce, 
            nhead=nhead, 
            nencoder=nencoder, 
            ndecoder=ndecoder, 
            lang_model=lang_model
        )
        warnings.warn("Model initialized...but no registered vocabulary! ")

        if not finetune_clip:
            self.predictor.encoders.vision_encoder.requires_grad_(False)
            self.predictor.encoders.vision_projector.requires_grad_(False)

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
        return self.predictor.reduces[0].weight.device
    
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
        self.unlabel_idx = self.label_embeddings.shape[0]-1
    
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

    def forward(self, batched_inputs):
        images = [input['image'].to(self.device, dtype=torch.float32) for input in batched_inputs]
        
        # Normalization
        pixels = [(image / 255 - self.pixel_mean.to(self.device)) / self.pixel_std.to(self.device) for image in images]
        pixels = ImageList.from_tensors(pixels, self.size_divisibility)

        # Resizing
        pixels_resized = F.interpolate(
            pixels.tensor, size=self.resolution, mode='bilinear', align_corners=False
        )

        if not self.training:
            return self.inference(pixels_resized, (batched_inputs[0]['height'], batched_inputs[0]['width']))
        
        # Ground Truth
        ids = [input['instances'].get_fields()['gt_classes'][:20] for input in batched_inputs]
        ids = torch.stack([F.pad(id, (0, 20-id.shape[0]), "constant", self.ignore_value) for id in ids])
        ids[ids > self.ignore_value] = self.ignore_value
        ids = ids.to(self.device, dtype=torch.long)

        masks = [input['instances'].get_fields()['gt_masks'][:20] for input in batched_inputs]
        masks = torch.stack([torch.concat([mask, torch.full((20-mask.shape[0], *mask.shape[1:]), 0, dtype=bool)]) for mask in masks]) # B L W H
        masks = masks.to(self.device, dtype=torch.float32)

        # Prediction
        mask_logits, pred_embeddings = self.predictor(pixels_resized)
        pred_embeddings = F.normalize(pred_embeddings, dim=-1)
        label_logits = pred_embeddings @ self.label_embeddings.T / self.temp
        mask_logits = F.interpolate(mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # Loss
        l1, l2, l3 = self.criterion(mask_logits, label_logits, masks, ids)

        loss = {"BCE": l1, "Dice": l2, "CLIP": l3}

        return loss